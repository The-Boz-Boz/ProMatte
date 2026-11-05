#include <doctest.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <thread>

#include "inference/backend_registry.h"
#include "inference/inference_worker.h"
#include "models/model_manager.h"
#include "preprocessing/preprocess.h"
#include "utils/file_utils.h"

using namespace promatte;

namespace {

bool loadModel(ModelDescriptor &out, std::string &path, const char *id)
{
	ModelManager mm(PROMATTE_MODELS_DIR, PROMATTE_MODELS_DIR);
	std::string err;
	if (!mm.loadManifestFile(fs::joinPath(PROMATTE_SOURCE_DIR, "data/models/manifest.json"), err))
		return false;
	auto d = mm.find(id);
	if (!d)
		return false;
	path = mm.resolvePath(id);
	if (path.empty())
		return false;
	out = *d;
	return true;
}

void fillFrame(FrameBuffer &f, uint32_t w, uint32_t h, uint8_t v)
{
	f.resize(w, h);
	std::fill(f.bgra.begin(), f.bgra.end(), v);
	for (size_t i = 3; i < f.bgra.size(); i += 4)
		f.bgra[i] = 255;
}

bool waitFor(const std::function<bool()> &pred, int ms)
{
	auto end = std::chrono::steady_clock::now() + std::chrono::milliseconds(ms);
	while (std::chrono::steady_clock::now() < end) {
		if (pred())
			return true;
		std::this_thread::sleep_for(std::chrono::milliseconds(5));
	}
	return pred();
}

} // namespace

TEST_CASE("worker without a model reports NoModel and never blocks submit")
{
	InferenceWorker w;
	w.start();
	WorkerConfig cfg; // empty model
	w.setConfig(cfg);
	FrameBuffer f;
	fillFrame(f, 64, 64, 100);
	for (int i = 0; i < 10; ++i)
		w.submitFrame(f); // must not block or crash
	CHECK(waitFor([&] { return w.status().state == WorkerState::NoModel; }, 2000));
	MatteBuffer m;
	CHECK_FALSE(w.takeMatte(m));
	w.stop();
	CHECK(w.status().state == WorkerState::Stopped);
}

TEST_CASE("worker with a missing model file reports an error, not a crash")
{
	InferenceWorker w;
	w.start();
	WorkerConfig cfg;
	cfg.model.id = "ghost";
	cfg.model.tiers.push_back({64, 64, 1.f, "Fixed"});
	cfg.modelPath = "C:/definitely/not/here.onnx";
	cfg.backend.kind = BackendKind::CPU;
	w.setConfig(cfg);
	CHECK(waitFor([&] { return w.status().state == WorkerState::Error; }, 5000));
	CHECK_FALSE(w.status().error.empty());
	w.stop();
}

TEST_CASE("worker processes frames with a real model on the CPU backend (latest frame wins)")
{
	ModelDescriptor model;
	std::string path;
	if (!loadModel(model, path, "mediapipe_selfie_landscape")) {
		MESSAGE("model not converted yet; skipping");
		return;
	}
	InferenceWorker w;
	w.start();
	WorkerConfig cfg;
	cfg.model = model;
	cfg.modelPath = path;
	cfg.backend.kind = BackendKind::CPU;
	cfg.backend.cpuThreads = 2;
	cfg.sourceWidth = 640;
	cfg.sourceHeight = 360;
	cfg.sourceFps = 30;
	w.setConfig(cfg);
	REQUIRE(waitFor([&] { return w.status().state == WorkerState::Running; }, 15000));
	uint32_t aw = 0, ah = 0;
	w.desiredResolution(aw, ah);
	CHECK(aw == 256);
	CHECK(ah == 144);

	// Flood the worker faster than it can process: no unbounded queue, drops counted.
	FrameBuffer f;
	for (int i = 0; i < 50; ++i) {
		fillFrame(f, aw, ah, uint8_t(i * 5));
		w.submitFrame(f);
	}
	MatteBuffer m;
	CHECK(waitFor([&] { return w.takeMatte(m); }, 10000));
	CHECK(m.width == aw);
	CHECK(m.height == ah);
	CHECK(m.rgba.size() == size_t(aw) * ah * 4);
	auto s = w.stats();
	CHECK(s.framesSubmitted == 50);
	CHECK(s.framesDropped > 0);
	CHECK(s.framesDropped < 50);
	CHECK(s.queueDepth <= 1);

	// A frame at the wrong size is resized on the CPU rather than rejected.
	fillFrame(f, 320, 200, 50);
	w.submitFrame(f);
	MatteBuffer m2;
	m2.seq = m.seq;
	CHECK(waitFor([&] { return w.takeMatte(m2); }, 5000));
	CHECK(m2.width == aw);

	// Disabling stops processing immediately.
	w.setEnabled(false);
	std::this_thread::sleep_for(std::chrono::milliseconds(300)); // let an in-flight frame finish
	auto before = w.stats().framesProcessed;
	for (int i = 0; i < 5; ++i) {
		fillFrame(f, aw, ah, 30);
		w.submitFrame(f);
		std::this_thread::sleep_for(std::chrono::milliseconds(20));
	}
	std::this_thread::sleep_for(std::chrono::milliseconds(200));
	CHECK(w.stats().framesProcessed == before);
	w.setEnabled(true);

	// Release the backend and re-configure -> works again.
	w.releaseBackend();
	CHECK(waitFor([&] { return w.status().state == WorkerState::Idle; }, 3000));
	w.setConfig(cfg);
	REQUIRE(waitFor([&] { return w.status().state == WorkerState::Running; }, 15000));
	w.stop();
}

TEST_CASE("RVM FP16 and FP32 exports agree on the CPU backend")
{
	ModelDescriptor m32, m16;
	std::string p32, p16;
	if (!loadModel(m32, p32, "rvm_mobilenetv3") || !loadModel(m16, p16, "rvm_mobilenetv3_fp16")) {
		MESSAGE("RVM models not downloaded; skipping");
		return;
	}
	CHECK(m16.variantOf == "rvm_mobilenetv3");
	CHECK(m16.precision == "fp16");
	const uint32_t w = 256, h = 160;
	// Smooth synthetic frame: a bright blob in the middle of a dark ramp.
	std::vector<float> src(size_t(3) * w * h);
	for (uint32_t y = 0; y < h; ++y)
		for (uint32_t x = 0; x < w; ++x) {
			const float dx = (float(x) - w / 2.f) / w, dy = (float(y) - h / 2.f) / h;
			const float v = dx * dx + dy * dy < 0.04f ? 0.8f : 0.1f + 0.2f * float(x) / w;
			for (int c = 0; c < 3; ++c)
				src[size_t(c) * w * h + size_t(y) * w + x] = v;
		}
	InferenceInput in;
	in.data = src.data();
	in.width = w;
	in.height = h;
	in.shape = {1, 3, int64_t(h), int64_t(w)};
	in.downsampleRatio = 0.5f;
	BackendConfig bc;
	bc.kind = BackendKind::CPU;
	bc.cpuThreads = 2;

	InferenceOutput o32, o16;
	std::string err;
	auto b32 = createBackend(BackendKind::CPU);
	REQUIRE_MESSAGE(b32->initialize(bc, m32, p32, err), err);
	CHECK_FALSE(b32->capabilities().fp16);
	auto b16 = createBackend(BackendKind::CPU);
	REQUIRE_MESSAGE(b16->initialize(bc, m16, p16, err), err);
	CHECK(b16->capabilities().fp16);
	// A few frames so the recurrent state (float16 on the FP16 model) is exercised.
	for (int i = 0; i < 3; ++i) {
		REQUIRE_MESSAGE(b32->processFrame(in, o32, err), err);
		REQUIRE_MESSAGE(b16->processFrame(in, o16, err), err);
	}
	REQUIRE(o16.alpha.size() == o32.alpha.size());
	REQUIRE(o16.foreground.size() == o32.foreground.size());
	double diff = 0, maxDiff = 0;
	for (size_t i = 0; i < o32.alpha.size(); ++i) {
		const double d = std::abs(double(o32.alpha[i]) - double(o16.alpha[i]));
		diff += d;
		maxDiff = std::max(maxDiff, d);
	}
	diff /= double(o32.alpha.size());
	double mean32 = 0, mean16 = 0;
	for (size_t i = 0; i < o32.alpha.size(); ++i) {
		mean32 += o32.alpha[i];
		mean16 += o16.alpha[i];
	}
	MESSAGE("RVM fp16 vs fp32 alpha: mean |d| " << diff << ", max " << maxDiff << ", alpha mean "
		<< mean32 / double(o32.alpha.size()) << " / " << mean16 / double(o16.alpha.size()));
	CHECK(diff < 0.01);
	// A synthetic frame has no person, so alpha is ~0 on both; the predicted
	// foreground colour (~ the input) exercises the float16 output path.
	REQUIRE(o32.foreground.size() == size_t(3) * w * h);
	double fgDiff = 0, fgMean = 0;
	for (size_t i = 0; i < o32.foreground.size(); ++i) {
		fgDiff += std::abs(double(o32.foreground[i]) - double(o16.foreground[i]));
		fgMean += o16.foreground[i];
	}
	fgDiff /= double(o32.foreground.size());
	fgMean /= double(o16.foreground.size());
	MESSAGE("RVM fp16 vs fp32 foreground: mean |d| " << fgDiff << ", mean " << fgMean);
	CHECK(fgMean > 0.05);
	CHECK(fgDiff < 0.01);

	// float16 input supplied directly (the worker's path) gives the same result.
	std::vector<uint16_t> half(src.size());
	for (size_t i = 0; i < src.size(); ++i)
		half[i] = floatToHalf(src[i]);
	InferenceInput inHalf = in;
	inHalf.data = nullptr;
	inHalf.dataHalf = half.data();
	InferenceOutput o16b;
	b16->resetState();
	for (int i = 0; i < 3; ++i)
		REQUIRE_MESSAGE(b16->processFrame(inHalf, o16b, err), err);
	REQUIRE(o16b.alpha.size() == o16.alpha.size());
	for (size_t i = 0; i < o16.alpha.size(); i += 97)
		CHECK(o16b.alpha[i] == doctest::Approx(o16.alpha[i]).epsilon(1e-3));
}
