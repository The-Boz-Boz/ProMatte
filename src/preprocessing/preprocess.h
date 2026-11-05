#pragma once
#include <cstddef>
#include <cstdint>

#include "inference/model_descriptor.h"

namespace promatte {

struct PreprocessOptions {
	TensorLayout layout = TensorLayout::NCHW;
	ChannelOrder order = ChannelOrder::RGB;
	float mean[3] = {0.f, 0.f, 0.f};
	float std[3] = {1.f, 1.f, 1.f};
};

// BGRA8 -> float tensor (3*w*h) with normalisation value = (x/255 - mean) / std.
void bgraToTensor(const uint8_t *bgra, uint32_t w, uint32_t h, uint32_t stride, const PreprocessOptions &opt,
		  float *out);

// IEEE 754 binary16 ("half") conversion, round-to-nearest-even. Kept here
// rather than taken from ONNX Runtime so the preprocessing layer stays free of
// runtime dependencies. Values are raw half bits.
uint16_t floatToHalf(float v);
float halfToFloat(uint16_t h);
// Table-driven bulk conversion half -> float (model outputs).
void halfToFloatN(const uint16_t *in, float *out, size_t n);

// Same as bgraToTensor, but writes float16 bits (for FP16 models such as
// rvm_mobilenetv3_fp16). Rounding happens once per channel value in the LUT.
void bgraToTensorHalf(const uint8_t *bgra, uint32_t w, uint32_t h, uint32_t stride, const PreprocessOptions &opt,
		      uint16_t *out);

// BGRA8 -> 8-bit luma (BT.601), used for motion estimation.
void bgraToLuma(const uint8_t *bgra, uint32_t w, uint32_t h, uint32_t stride, uint8_t *luma);

// Motion estimation for skipping inference on static frames. Each of the tw x th
// thumbnail cells is the mean luma (BT.601) of the BGRA pixels it covers,
// sampled every second pixel in both directions: cheap, and averaging cuts
// camera noise so that a still scene measures well below a real movement.
void lumaThumbnail(const uint8_t *bgra, uint32_t w, uint32_t h, uint32_t stride, uint32_t tw, uint32_t th,
		   uint8_t *out);
// Mean absolute difference of two luma buffers of n samples, in luma levels (0..255).
double meanAbsDiff(const uint8_t *a, const uint8_t *b, size_t n);

// Static-scene decision on two luma thumbnails (lumaThumbnail). A mean over the
// whole thumbnail hides local motion (a hand is 1-2% of the cells), so the test
// is local: a frame moves when enough cells, or a 4x4-cell block, change by more
// than the frame's own noise.
//   noise   = median |delta| over all cells, capped at kMotionNoiseCap. Sensor
//             noise and a uniform brightness step (auto exposure) shift every
//             cell alike and land in the median; motion of a minority does not.
//   changed = cells with |delta| >= cellDelta + 2 * noise
//   block   = mean of min(max(|delta| - noise, 0), cell threshold) over a 4x4
//             block of cells; catches low-contrast change spread over an area
//             (hand over a similar background, a large brightness step). The
//             per-cell clip keeps one bright cell from tripping it on its own.
// The frame is static when changed < minChangedCells and every block mean is
// below max(cellDelta / 2, 1.5).
struct MotionSkipParams {
	float cellDelta = 6.f;   // levels above the noise for a cell to count as changed
	int minChangedCells = 6; // this many changed cells (of 64x36 = 2304) mean motion
};

struct MotionMeasure {
	float noise = 0;       // estimated noise floor, levels (capped median |delta|)
	int changedCells = 0;  // cells over the cell threshold
	float maxBlock = 0;    // largest clipped block mean, levels
	bool moving = false;
};

constexpr float kMotionNoiseCap = 4.f;
constexpr uint32_t kMotionBlockCells = 4; // block = 4x4 thumbnail cells

MotionMeasure measureMotion(const uint8_t *cur, const uint8_t *ref, uint32_t tw, uint32_t th,
			    const MotionSkipParams &p);

// UI "motion sensitivity" 0..100 (default 50) -> detector parameters. Each 35
// points halve (or double) both the cell threshold and the changed-cell count:
// 0 -> 16 levels / 16 cells, 50 -> 6 / 6, 100 -> ~2.2 / 2.
constexpr int kDefaultMotionSensitivity = 50;
MotionSkipParams motionSkipParamsForSensitivity(int sensitivity);
// Settings migration: the old mean-|delta| threshold (default 1.5 levels) mapped
// onto the sensitivity scale on the same log curve: 1.5 -> 50, 0.75 -> 85,
// 3 -> 15, <= 0.5 (or 0 = "only identical frames") -> 100.
int motionSensitivityFromLegacyThreshold(double meanLevels);

// Area-averaging (box) downscale for BGRA8; falls back to bilinear when upscaling.
void resizeBgra(const uint8_t *src, uint32_t sw, uint32_t sh, uint32_t sstride, uint8_t *dst, uint32_t dw, uint32_t dh);

// Chooses the AI resolution for a source of sw x sh given a tier: keeps the tier's
// pixel budget while matching the source aspect ratio and alignment.
void chooseAiResolution(uint32_t sw, uint32_t sh, uint32_t tierW, uint32_t tierH, bool fixedInput, uint32_t alignment,
			uint32_t &outW, uint32_t &outH);

} // namespace promatte
