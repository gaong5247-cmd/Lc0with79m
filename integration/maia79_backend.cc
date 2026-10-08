/*
 * Maia3-79M inference backend for upstream LC0.
 * LC0 portions: GPL-3.0-or-later; integration: AGPL-3.0-or-later.
 * Uses the pretrained original Maia3 weights exported as a float32 ONNX graph.
 */
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "chess/position.h"
#include "neural/backend.h"
#include "neural/register.h"
#include "neural/shared_params.h"
#include "utils/commandline.h"
#include "utils/optionsdict.h"
#include "onnxruntime_cxx_api.h"

namespace lczero {
namespace {

constexpr int kHistory = 8;
constexpr int kSquares = 64;
constexpr int kChannels = kHistory * 12 + 1;  // 97, final channel is clk_ponder (unused).
constexpr int kPolicy = 4352;
using Tokens = std::array<float, kSquares * kChannels>;

int PieceChannel(const ChessBoard& board, const uint64_t square_mask) {
  const bool own = (board.ours().as_int() & square_mask) != 0;
  const bool theirs = (board.theirs().as_int() & square_mask) != 0;
  if (!own && !theirs) return -1;
  const int side = own ? 0 : 6;
  if (board.pawns().as_int() & square_mask) return side;
  if (board.knights().as_int() & square_mask) return side + 1;
  if (board.bishops().as_int() & square_mask) return side + 2;
  if (board.rooks().as_int() & square_mask) return side + 3;
  if (board.queens().as_int() & square_mask) return side + 4;
  if (board.kings().as_int() & square_mask) return side + 5;
  throw std::runtime_error("Maia: unknown piece in input position");
}

// Maia's dataset.tokenize_board() rotates each historical board into the
// perspective of the player to move AT THAT HISTORICAL PLY, not the root's.
// LC0 ChessBoard::GetBoard() is already mirrored for black to move.
Tokens Tokenize(std::span<const Position> history) {
  if (history.empty()) throw std::runtime_error("Maia: empty history");
  Tokens tokens{};
  const int count = static_cast<int>(history.size());
  for (int h = 0; h < kHistory; ++h) {
    // Left-pad with the oldest available position (official Maia convention).
    const int idx = std::clamp(count - kHistory + h, 0, count - 1);
    const ChessBoard& b = history[idx].GetBoard();
    for (int sq = 0; sq < kSquares; ++sq) {
      const int piece = PieceChannel(b, uint64_t{1} << sq);
      if (piece >= 0) tokens[sq * kChannels + h * 12 + piece] = 1.0f;
    }
  }
  return tokens;
}

// Maia move vocabulary: 64*64 normal from-to indices, followed by 8*8*4
// promotions ordered (from_file, to_file, q/r/b/n). Both LC0 Move and Maia
// use perspective-relative squares; LC0 encodes castling as king-takes-rook.
int MaiaMoveIndex(Move move) {
  int from = move.from().as_idx();
  int to = move.to().as_idx();
  if (move.is_promotion()) {
    const PieceType p = move.promotion();
    int promo = p == kQueen ? 0 : p == kRook ? 1 : p == kBishop ? 2
                             : p == kKnight ? 3 : -1;
    if (promo < 0) throw std::runtime_error("Maia: invalid promotion piece");
    return 4096 + ((from % 8) * 8 + (to % 8)) * 4 + promo;
  }
  if (move.is_castling()) {
    const int side = (to % 8 > from % 8) ? 6 : 2;
    to = (from / 8) * 8 + side;
  }
  return from * 64 + to;
}

struct Model {
  Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "lc0-maia3"};
  Ort::Session session{nullptr};
  int64_t self_elo = 2600;
  int64_t oppo_elo = 2600;
  float temperature = 1.0f;

  Model(const std::string& path, int self, int oppo, float temp)
      : self_elo(std::clamp(self, 0, 5000)),
        oppo_elo(std::clamp(oppo, 0, 5000)),
        temperature(temp) {
    if (!(temperature > 0.0f) || !std::isfinite(temperature))
      throw std::runtime_error("Maia temperature must be finite and positive");
    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(1);
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    auto model_file = std::filesystem::path(path);
    if (model_file.is_relative() && !std::filesystem::exists(model_file)) {
      model_file = std::filesystem::path(CommandLine::BinaryDirectory()) / model_file;
    }
    const auto native = model_file.native();
    session = Ort::Session(env, native.c_str(), opts);
    if (session.GetInputCount() != 3 || session.GetOutputCount() < 2)
      throw std::runtime_error("Maia ONNX must have 3 inputs and >=2 outputs");
  }
};

struct Pending {
  Tokens tokens;
  std::vector<int> moves;
  EvalResultPtr dest;
};

class MaiaComputation final : public BackendComputation {
 public:
  explicit MaiaComputation(std::shared_ptr<Model> model) : model_(std::move(model)) {}

  size_t UsedBatchSize() const override { return pending_.size(); }

  AddInputResult AddInput(const EvalPosition& pos, EvalResultPtr result) override {
    if (result.p.size() != pos.legal_moves.size())
      throw std::runtime_error("Maia: legal move/output size mismatch");
    Pending task;
    task.tokens = Tokenize(pos.pos);
    task.dest = result;
    task.moves.reserve(pos.legal_moves.size());
    for (Move move : pos.legal_moves) task.moves.push_back(MaiaMoveIndex(move));
    pending_.push_back(std::move(task));
    return ENQUEUED_FOR_EVAL;
  }

  void ComputeBlocking() override {
    constexpr std::array<int64_t, 3> token_shape{1, 64, 97};
    constexpr std::array<int64_t, 1> elo_shape{1};
    constexpr const char* input_names[] = {"tokens", "self_elos", "oppo_elos"};
    constexpr const char* output_names[] = {"policy", "value"};
    auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    for (auto& task : pending_) {
      auto self = model_->self_elo;
      auto oppo = model_->oppo_elo;
      std::array<Ort::Value, 3> inputs{
        Ort::Value::CreateTensor<float>(memory, task.tokens.data(), task.tokens.size(),
                                        token_shape.data(), token_shape.size()),
        Ort::Value::CreateTensor<int64_t>(memory, &self, 1, elo_shape.data(), elo_shape.size()),
        Ort::Value::CreateTensor<int64_t>(memory, &oppo, 1, elo_shape.data(), elo_shape.size())
      };
      auto outputs = model_->session.Run(Ort::RunOptions{nullptr}, input_names,
                                         inputs.data(), inputs.size(),
                                         output_names, 2);
      auto p_info = outputs[0].GetTensorTypeAndShapeInfo();
      auto v_info = outputs[1].GetTensorTypeAndShapeInfo();
      if (p_info.GetElementCount() != kPolicy || v_info.GetElementCount() != 3 ||
          p_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
          v_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
        throw std::runtime_error("Maia ONNX output mismatch: expected float32 [1,4352], [1,3]");
      const float* logits = outputs[0].GetTensorData<float>();
      const float* value = outputs[1].GetTensorData<float>();

      // All returned policy numbers are normalized *over legal moves*.
      if (!task.moves.empty()) {
        float maximum = -std::numeric_limits<float>::infinity();
        for (int index : task.moves)
          maximum = std::max(maximum, logits[index] / model_->temperature);
        double sum = 0.0;
        for (size_t i = 0; i < task.moves.size(); ++i) {
          const double p = std::exp(double(logits[task.moves[i]] / model_->temperature - maximum));
          task.dest.p[i] = static_cast<float>(p);
          sum += p;
        }
        if (sum <= 0.0 || !std::isfinite(sum))
          throw std::runtime_error("Maia returned invalid policy probabilities");
        for (float& p : task.dest.p) p = static_cast<float>(p / sum);
      }

      // Value head is W/D/L from the current side-to-move's perspective.
      const float vmax = std::max({value[0], value[1], value[2]});
      // Maia3 labels value logits [loss, draw, win], unlike UCI [W,D,L].
      const double w = std::exp(double(value[2] - vmax));
      const double d = std::exp(double(value[1] - vmax));
      const double l = std::exp(double(value[0] - vmax));
      const double denom = w + d + l;
      if (!std::isfinite(denom) || denom <= 0.0)
        throw std::runtime_error("Maia returned invalid WDL probabilities");
      if (task.dest.q) *task.dest.q = static_cast<float>((w - l) / denom);
      if (task.dest.d) *task.dest.d = static_cast<float>(d / denom);
      if (task.dest.m) *task.dest.m = 0.0f;
    }
    pending_.clear();
  }
 private:
  std::shared_ptr<Model> model_;
  std::vector<Pending> pending_;
};

class MaiaBackend final : public Backend {
 public:
  MaiaBackend(const std::string& path, int self, int oppo, float temp)
      : model_(std::make_shared<Model>(path, self, oppo, temp)) {}

  BackendAttributes GetAttributes() const override {
    return BackendAttributes{.has_mlh = false, .has_wdl = true, .runs_on_cpu = true,
      .suggested_num_search_threads = 1, .recommended_batch_size = 1,
      .maximum_batch_size = 1};
  }

  std::unique_ptr<BackendComputation> CreateComputation() override {
    return std::make_unique<MaiaComputation>(model_);
  }

 private:
  std::shared_ptr<Model> model_;
};

class MaiaFactory final : public BackendFactory {
 public:
  int GetPriority() const override { return -100; }
  std::string_view GetName() const override { return "maia79"; }

  std::unique_ptr<Backend> Create(const OptionsDict& opts) override {
    OptionsDict backend_options(&opts);
    const auto raw = opts.Get<std::string>(SharedBackendParams::kBackendOptionsId);
    if (!raw.empty()) backend_options.AddSubdictFromString(raw);
    const auto model = backend_options.GetOrDefault<std::string>("model", "maia3-79m.onnx");
    const int self = backend_options.GetOrDefault<int>("selfelo", 2600);
    const int oppo = backend_options.GetOrDefault<int>("oppoelo", 2600);
    const float temp = backend_options.GetOrDefault<float>("temperature", 1.0f);
    return std::make_unique<MaiaBackend>(model, self, oppo, temp);
  }
};
[[maybe_unused]] const BackendManager::Register maia_registration{
    std::make_unique<MaiaFactory>()};

}  // namespace
}  // namespace lczero
