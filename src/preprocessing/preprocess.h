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

// Area-averaging (box) downscale for BGRA8; falls back to bilinear when upscaling.
void resizeBgra(const uint8_t *src, uint32_t sw, uint32_t sh, uint32_t sstride, uint8_t *dst, uint32_t dw, uint32_t dh);

// Chooses the AI resolution for a source of sw x sh given a tier: keeps the tier's
// pixel budget while matching the source aspect ratio and alignment.
void chooseAiResolution(uint32_t sw, uint32_t sh, uint32_t tierW, uint32_t tierH, bool fixedInput, uint32_t alignment,
			uint32_t &outW, uint32_t &outH);

} // namespace promatte
