#include "preprocessing/preprocess.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

namespace promatte {

namespace {

// Writes one BGRA8 frame through per-channel lookup tables into a planar (NCHW)
// or interleaved (NHWC) tensor; T is float or raw float16 bits.
template<typename T>
void bgraThroughLut(const uint8_t *bgra, uint32_t w, uint32_t h, uint32_t stride, const PreprocessOptions &opt,
		    const T (&lut)[3][256], T *out)
{
	const size_t n = size_t(w) * h;
	// Source is BGRA: index 0 = B, 1 = G, 2 = R. Map to tensor channel order.
	const int srcIdx[3] = {opt.order == ChannelOrder::RGB ? 2 : 0, 1, opt.order == ChannelOrder::RGB ? 0 : 2};

	if (opt.layout == TensorLayout::NCHW) {
		T *c0 = out, *c1 = out + n, *c2 = out + 2 * n;
		for (uint32_t y = 0; y < h; ++y) {
			const uint8_t *row = bgra + size_t(y) * stride;
			size_t base = size_t(y) * w;
			for (uint32_t x = 0; x < w; ++x) {
				const uint8_t *p = row + size_t(x) * 4;
				c0[base + x] = lut[0][p[srcIdx[0]]];
				c1[base + x] = lut[1][p[srcIdx[1]]];
				c2[base + x] = lut[2][p[srcIdx[2]]];
			}
		}
	} else {
		for (uint32_t y = 0; y < h; ++y) {
			const uint8_t *row = bgra + size_t(y) * stride;
			T *o = out + size_t(y) * w * 3;
			for (uint32_t x = 0; x < w; ++x) {
				const uint8_t *p = row + size_t(x) * 4;
				o[x * 3 + 0] = lut[0][p[srcIdx[0]]];
				o[x * 3 + 1] = lut[1][p[srcIdx[1]]];
				o[x * 3 + 2] = lut[2][p[srcIdx[2]]];
			}
		}
	}
}

void buildFloatLut(const PreprocessOptions &opt, float (&lut)[3][256])
{
	// Precompute per-channel lookup tables (256 entries) — avoids per-pixel divides.
	for (int c = 0; c < 3; ++c) {
		float inv = opt.std[c] != 0.f ? 1.f / opt.std[c] : 1.f;
		for (int v = 0; v < 256; ++v)
			lut[c][v] = (float(v) / 255.f - opt.mean[c]) * inv;
	}
}

} // namespace

uint16_t floatToHalf(float v)
{
	uint32_t f;
	std::memcpy(&f, &v, sizeof(f));
	const uint32_t sign = (f >> 16) & 0x8000u;
	const uint32_t absf = f & 0x7FFFFFFFu;
	if (absf >= 0x7F800000u) // Inf or NaN (keep NaN quiet and non-zero)
		return uint16_t(sign | 0x7C00u | (absf > 0x7F800000u ? 0x0200u | ((absf >> 13) & 0x03FFu) : 0u));
	if (absf >= 0x477FF000u) // rounds to >= 65520: overflow to Inf
		return uint16_t(sign | 0x7C00u);
	if (absf < 0x38800000u) { // below the smallest normal half (2^-14): subnormal or zero
		if (absf < 0x33000000u) // < 2^-25: rounds to zero
			return uint16_t(sign);
		const uint32_t mant = (absf & 0x007FFFFFu) | 0x00800000u;
		const int shift = 113 - int(absf >> 23) + 13; // 126 - exp, plus the 13 dropped mantissa bits
		uint32_t half = mant >> shift;
		const uint32_t rem = mant & ((1u << shift) - 1u);
		const uint32_t halfway = 1u << (shift - 1);
		if (rem > halfway || (rem == halfway && (half & 1u)))
			++half;
		return uint16_t(sign | half);
	}
	// Normal: rebias the exponent (127 -> 15) and round the mantissa to 10 bits.
	uint32_t half = ((absf >> 13) - (112u << 10));
	const uint32_t rem = absf & 0x1FFFu;
	if (rem > 0x1000u || (rem == 0x1000u && (half & 1u)))
		++half; // a carry into the exponent is the correct result
	return uint16_t(sign | half);
}

float halfToFloat(uint16_t h)
{
	const uint32_t sign = uint32_t(h & 0x8000u) << 16;
	const uint32_t exp = (h >> 10) & 0x1Fu;
	uint32_t mant = h & 0x03FFu;
	uint32_t f;
	if (exp == 0x1Fu) {
		f = sign | 0x7F800000u | (mant << 13);
	} else if (exp != 0) {
		f = sign | ((exp + 112u) << 23) | (mant << 13);
	} else if (mant == 0) {
		f = sign;
	} else {
		// Subnormal half: normalise into a float.
		uint32_t e = 113;
		while (!(mant & 0x0400u)) {
			mant <<= 1;
			--e;
		}
		f = sign | (e << 23) | ((mant & 0x03FFu) << 13);
	}
	float v;
	std::memcpy(&v, &f, sizeof(v));
	return v;
}

void halfToFloatN(const uint16_t *in, float *out, size_t n)
{
	// 256 KB table, built once (thread-safe static init).
	static const std::vector<float> table = [] {
		std::vector<float> t(65536);
		for (uint32_t i = 0; i < 65536; ++i)
			t[i] = halfToFloat(uint16_t(i));
		return t;
	}();
	const float *t = table.data();
	for (size_t i = 0; i < n; ++i)
		out[i] = t[in[i]];
}

void bgraToTensor(const uint8_t *bgra, uint32_t w, uint32_t h, uint32_t stride, const PreprocessOptions &opt, float *out)
{
	float lut[3][256];
	buildFloatLut(opt, lut);
	bgraThroughLut(bgra, w, h, stride, opt, lut, out);
}

void bgraToTensorHalf(const uint8_t *bgra, uint32_t w, uint32_t h, uint32_t stride, const PreprocessOptions &opt,
		      uint16_t *out)
{
	float lut[3][256];
	buildFloatLut(opt, lut);
	uint16_t halfLut[3][256];
	for (int c = 0; c < 3; ++c)
		for (int v = 0; v < 256; ++v)
			halfLut[c][v] = floatToHalf(lut[c][v]);
	bgraThroughLut(bgra, w, h, stride, opt, halfLut, out);
}

void bgraToLuma(const uint8_t *bgra, uint32_t w, uint32_t h, uint32_t stride, uint8_t *luma)
{
	for (uint32_t y = 0; y < h; ++y) {
		const uint8_t *row = bgra + size_t(y) * stride;
		uint8_t *o = luma + size_t(y) * w;
		for (uint32_t x = 0; x < w; ++x) {
			const uint8_t *p = row + size_t(x) * 4;
			// BT.601 integer approximation: (77R + 150G + 29B) >> 8
			o[x] = uint8_t((77u * p[2] + 150u * p[1] + 29u * p[0]) >> 8);
		}
	}
}

void lumaThumbnail(const uint8_t *bgra, uint32_t w, uint32_t h, uint32_t stride, uint32_t tw, uint32_t th,
		   uint8_t *out)
{
	if (tw == 0 || th == 0)
		return;
	if (w == 0 || h == 0) {
		std::fill_n(out, size_t(tw) * th, uint8_t(0));
		return;
	}
	for (uint32_t ty = 0; ty < th; ++ty) {
		const uint32_t y0 = uint32_t(uint64_t(ty) * h / th);
		const uint32_t y1 = std::max(y0 + 1, uint32_t(uint64_t(ty + 1) * h / th));
		for (uint32_t tx = 0; tx < tw; ++tx) {
			const uint32_t x0 = uint32_t(uint64_t(tx) * w / tw);
			const uint32_t x1 = std::max(x0 + 1, uint32_t(uint64_t(tx + 1) * w / tw));
			uint32_t acc = 0, cnt = 0;
			for (uint32_t y = y0; y < y1; y += 2) {
				const uint8_t *row = bgra + size_t(y) * stride;
				for (uint32_t x = x0; x < x1; x += 2) {
					const uint8_t *p = row + size_t(x) * 4;
					acc += (77u * p[2] + 150u * p[1] + 29u * p[0]) >> 8;
					++cnt;
				}
			}
			out[size_t(ty) * tw + tx] = uint8_t((acc + cnt / 2) / cnt);
		}
	}
}

double meanAbsDiff(const uint8_t *a, const uint8_t *b, size_t n)
{
	if (n == 0)
		return 0.0;
	uint64_t sum = 0;
	for (size_t i = 0; i < n; ++i)
		sum += uint64_t(a[i] > b[i] ? a[i] - b[i] : b[i] - a[i]);
	return double(sum) / double(n);
}

MotionMeasure measureMotion(const uint8_t *cur, const uint8_t *ref, uint32_t tw, uint32_t th,
			    const MotionSkipParams &p)
{
	MotionMeasure m;
	const size_t n = size_t(tw) * th;
	if (n == 0)
		return m;
	uint32_t hist[256] = {};
	for (size_t i = 0; i < n; ++i)
		++hist[cur[i] > ref[i] ? cur[i] - ref[i] : ref[i] - cur[i]];
	// Median |delta|: the noise floor shared by most cells.
	size_t acc = 0;
	int median = 0;
	for (; median < 255; ++median) {
		acc += hist[median];
		if (acc * 2 > n)
			break;
	}
	m.noise = std::min(float(median), kMotionNoiseCap);
	const float cellDelta = std::max(p.cellDelta, 0.f);
	const float cellThr = std::max(cellDelta + 2.f * m.noise, 1.f);
	const float blockThr = std::max(cellDelta * 0.5f, 1.5f);
	for (int d = 0; d < 256; ++d)
		if (float(d) >= cellThr)
			m.changedCells += int(hist[d]);

	for (uint32_t by = 0; by < th; by += kMotionBlockCells) {
		const uint32_t y1 = std::min(by + kMotionBlockCells, th);
		for (uint32_t bx = 0; bx < tw; bx += kMotionBlockCells) {
			const uint32_t x1 = std::min(bx + kMotionBlockCells, tw);
			float sum = 0;
			for (uint32_t y = by; y < y1; ++y)
				for (uint32_t x = bx; x < x1; ++x) {
					const size_t i = size_t(y) * tw + x;
					const float d = float(cur[i] > ref[i] ? cur[i] - ref[i] : ref[i] - cur[i]);
					sum += std::clamp(d - m.noise, 0.f, cellThr);
				}
			m.maxBlock = std::max(m.maxBlock, sum / float((y1 - by) * (x1 - bx)));
		}
	}
	m.moving = m.changedCells >= std::max(p.minChangedCells, 1) || m.maxBlock >= blockThr;
	return m;
}

MotionSkipParams motionSkipParamsForSensitivity(int sensitivity)
{
	const double s = std::clamp(sensitivity, 0, 100);
	const double scale = std::exp2((double(kDefaultMotionSensitivity) - s) / 35.0);
	MotionSkipParams p;
	p.cellDelta = float(6.0 * scale);
	p.minChangedCells = std::max(1, int(std::lround(6.0 * scale)));
	return p;
}

int motionSensitivityFromLegacyThreshold(double meanLevels)
{
	if (!(meanLevels > 0.0))
		return 100;
	const double s = double(kDefaultMotionSensitivity) + 35.0 * std::log2(1.5 / meanLevels);
	return int(std::clamp(std::lround(s), 0L, 100L));
}

void resizeBgra(const uint8_t *src, uint32_t sw, uint32_t sh, uint32_t sstride, uint8_t *dst, uint32_t dw, uint32_t dh)
{
	if (dw == 0 || dh == 0 || sw == 0 || sh == 0)
		return;
	if (dw == sw && dh == sh) {
		for (uint32_t y = 0; y < sh; ++y)
			std::copy_n(src + size_t(y) * sstride, size_t(sw) * 4, dst + size_t(y) * dw * 4);
		return;
	}
	const bool down = dw <= sw && dh <= sh;
	if (down) {
		// Box filter: average all source pixels covering each destination pixel.
		for (uint32_t y = 0; y < dh; ++y) {
			uint32_t y0 = uint32_t(uint64_t(y) * sh / dh);
			uint32_t y1 = std::max(y0 + 1, uint32_t(uint64_t(y + 1) * sh / dh));
			for (uint32_t x = 0; x < dw; ++x) {
				uint32_t x0 = uint32_t(uint64_t(x) * sw / dw);
				uint32_t x1 = std::max(x0 + 1, uint32_t(uint64_t(x + 1) * sw / dw));
				uint32_t acc[4] = {0, 0, 0, 0};
				uint32_t cnt = 0;
				for (uint32_t yy = y0; yy < y1; ++yy) {
					const uint8_t *row = src + size_t(yy) * sstride;
					for (uint32_t xx = x0; xx < x1; ++xx) {
						const uint8_t *p = row + size_t(xx) * 4;
						acc[0] += p[0];
						acc[1] += p[1];
						acc[2] += p[2];
						acc[3] += p[3];
						++cnt;
					}
				}
				uint8_t *o = dst + (size_t(y) * dw + x) * 4;
				for (int c = 0; c < 4; ++c)
					o[c] = uint8_t((acc[c] + cnt / 2) / cnt);
			}
		}
	} else {
		// Bilinear.
		for (uint32_t y = 0; y < dh; ++y) {
			float fy = (float(y) + 0.5f) * float(sh) / float(dh) - 0.5f;
			int y0 = std::clamp(int(std::floor(fy)), 0, int(sh) - 1);
			int y1 = std::min(y0 + 1, int(sh) - 1);
			float wy = std::clamp(fy - float(y0), 0.f, 1.f);
			for (uint32_t x = 0; x < dw; ++x) {
				float fx = (float(x) + 0.5f) * float(sw) / float(dw) - 0.5f;
				int x0 = std::clamp(int(std::floor(fx)), 0, int(sw) - 1);
				int x1 = std::min(x0 + 1, int(sw) - 1);
				float wx = std::clamp(fx - float(x0), 0.f, 1.f);
				const uint8_t *p00 = src + size_t(y0) * sstride + size_t(x0) * 4;
				const uint8_t *p01 = src + size_t(y0) * sstride + size_t(x1) * 4;
				const uint8_t *p10 = src + size_t(y1) * sstride + size_t(x0) * 4;
				const uint8_t *p11 = src + size_t(y1) * sstride + size_t(x1) * 4;
				uint8_t *o = dst + (size_t(y) * dw + x) * 4;
				for (int c = 0; c < 4; ++c) {
					float top = p00[c] + (p01[c] - p00[c]) * wx;
					float bot = p10[c] + (p11[c] - p10[c]) * wx;
					o[c] = uint8_t(std::clamp(top + (bot - top) * wy + 0.5f, 0.f, 255.f));
				}
			}
		}
	}
}

void chooseAiResolution(uint32_t sw, uint32_t sh, uint32_t tierW, uint32_t tierH, bool fixedInput, uint32_t alignment,
			uint32_t &outW, uint32_t &outH)
{
	if (fixedInput || sw == 0 || sh == 0) {
		outW = tierW;
		outH = tierH;
		return;
	}
	// Keep the tier's pixel budget, follow the source aspect ratio.
	double budget = double(tierW) * double(tierH);
	double aspect = double(sw) / double(sh);
	double h = std::sqrt(budget / aspect);
	double w = h * aspect;
	uint32_t al = std::max<uint32_t>(alignment, 1);
	auto align = [al](double v) {
		uint32_t r = uint32_t((v + al / 2.0) / al) * al;
		return std::max<uint32_t>(r, al);
	};
	outW = align(w);
	outH = align(h);
	// Never upscale beyond the source.
	if (outW > sw || outH > sh) {
		outW = std::max<uint32_t>(al, (sw / al) * al);
		outH = std::max<uint32_t>(al, (sh / al) * al);
	}
}

} // namespace promatte
