#include <doctest.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "preprocessing/preprocess.h"

using namespace promatte;

TEST_CASE("bgraToTensor NCHW RGB with normalisation")
{
	// one pixel: B=0, G=128, R=255
	uint8_t px[4] = {0, 128, 255, 255};
	PreprocessOptions o;
	o.layout = TensorLayout::NCHW;
	o.order = ChannelOrder::RGB;
	o.mean[0] = o.mean[1] = o.mean[2] = 0.5f;
	o.std[0] = o.std[1] = o.std[2] = 0.5f;
	float out[3];
	bgraToTensor(px, 1, 1, 4, o, out);
	CHECK(out[0] == doctest::Approx(1.0f));                    // R
	CHECK(out[1] == doctest::Approx((128 / 255.f - 0.5f) * 2)); // G
	CHECK(out[2] == doctest::Approx(-1.0f));                   // B
}

TEST_CASE("bgraToTensor NHWC BGR keeps interleaved order")
{
	uint8_t px[8] = {10, 20, 30, 255, 40, 50, 60, 255};
	PreprocessOptions o;
	o.layout = TensorLayout::NHWC;
	o.order = ChannelOrder::BGR;
	float out[6];
	bgraToTensor(px, 2, 1, 8, o, out);
	CHECK(out[0] == doctest::Approx(10 / 255.f));
	CHECK(out[1] == doctest::Approx(20 / 255.f));
	CHECK(out[2] == doctest::Approx(30 / 255.f));
	CHECK(out[3] == doctest::Approx(40 / 255.f));
}

TEST_CASE("luma conversion")
{
	uint8_t white[4] = {255, 255, 255, 255}, black[4] = {0, 0, 0, 255};
	uint8_t l;
	bgraToLuma(white, 1, 1, 4, &l);
	CHECK(l >= 254);
	bgraToLuma(black, 1, 1, 4, &l);
	CHECK(l == 0);
}

TEST_CASE("resizeBgra box downscale averages")
{
	// 2x2 -> 1x1
	uint8_t src[16] = {0, 0, 0, 255, 100, 100, 100, 255, 200, 200, 200, 255, 100, 100, 100, 255};
	uint8_t dst[4];
	resizeBgra(src, 2, 2, 8, dst, 1, 1);
	CHECK(dst[0] == 100);
	CHECK(dst[3] == 255);
}

TEST_CASE("resizeBgra upscale is bilinear-ish and in range")
{
	uint8_t src[8] = {0, 0, 0, 255, 200, 200, 200, 255};
	std::vector<uint8_t> dst(4 * 4);
	resizeBgra(src, 2, 1, 8, dst.data(), 4, 1);
	CHECK(dst[0] <= dst[4]);
	CHECK(dst[4] <= dst[8]);
	CHECK(dst[8] <= dst[12]);
}

TEST_CASE("chooseAiResolution keeps aspect and alignment for dynamic models")
{
	uint32_t w, h;
	chooseAiResolution(1920, 1080, 512, 288, false, 32, w, h);
	CHECK(w % 32 == 0);
	CHECK(h % 32 == 0);
	CHECK(double(w) / h == doctest::Approx(16.0 / 9.0).epsilon(0.15));
	CHECK(w * h <= 512 * 288 * 1.3);
	CHECK(w * h >= 512 * 288 * 0.7);

	// portrait source
	chooseAiResolution(720, 1280, 512, 288, false, 32, w, h);
	CHECK(h > w);

	// fixed model ignores the source
	chooseAiResolution(1920, 1080, 256, 144, true, 1, w, h);
	CHECK(w == 256);
	CHECK(h == 144);

	// never upscale beyond source
	chooseAiResolution(320, 180, 1280, 704, false, 32, w, h);
	CHECK(w <= 320);
	CHECK(h <= 180);
}

TEST_CASE("float16 conversion round-trips and handles the edge cases")
{
	// Exactly representable values survive both directions.
	const float exact[] = {0.f, 1.f, -1.f, 0.5f, 2.f, 1024.f, 65504.f, -65504.f, 6.103515625e-05f /* 2^-14 */,
			       5.9604644775390625e-08f /* 2^-24, smallest subnormal */, 0.333251953125f};
	for (float v : exact)
		CHECK(halfToFloat(floatToHalf(v)) == v);
	CHECK(floatToHalf(0.f) == 0x0000);
	CHECK(floatToHalf(-0.f) == 0x8000);
	CHECK(floatToHalf(1.f) == 0x3C00);
	CHECK(floatToHalf(65504.f) == 0x7BFF);
	CHECK(floatToHalf(1e6f) == 0x7C00);   // overflow -> +inf
	CHECK(floatToHalf(-1e6f) == 0xFC00);  // -inf
	CHECK(floatToHalf(65520.f) == 0x7C00); // halfway above the max rounds to inf
	CHECK(floatToHalf(1e-9f) == 0x0000);  // underflow -> 0
	CHECK(floatToHalf(5.9604644775390625e-08f) == 0x0001);
	const uint16_t nan = floatToHalf(std::numeric_limits<float>::quiet_NaN());
	CHECK((nan & 0x7C00) == 0x7C00);
	CHECK((nan & 0x03FF) != 0);
	CHECK(std::isinf(halfToFloat(0x7C00)));
	CHECK(std::isnan(halfToFloat(0x7E00)));
	// Round to nearest even: 1 + 2^-11 is halfway between 1 and 1 + 2^-10.
	CHECK(floatToHalf(1.f + 1.f / 2048.f) == 0x3C00);
	CHECK(floatToHalf(1.f + 3.f / 2048.f) == 0x3C02);
	// Every half value survives half -> float -> half (NaNs excluded).
	int bad = 0;
	for (uint32_t i = 0; i < 65536; ++i) {
		const uint16_t h = uint16_t(i);
		if ((h & 0x7C00) == 0x7C00 && (h & 0x03FF))
			continue;
		if (floatToHalf(halfToFloat(h)) != h)
			++bad;
	}
	CHECK(bad == 0);
	// Bulk conversion matches the scalar one.
	const uint16_t in[4] = {0x3C00, 0xC000, 0x0001, 0x7BFF};
	float out[4];
	halfToFloatN(in, out, 4);
	for (int i = 0; i < 4; ++i)
		CHECK(out[i] == halfToFloat(in[i]));
	// Relative error of normal numbers stays within half an ULP (2^-11).
	for (float v = 1e-4f; v < 60000.f; v *= 1.37f) {
		const float r = halfToFloat(floatToHalf(v));
		CHECK(std::fabs(r - v) <= v * (1.f / 2048.f));
	}
}

TEST_CASE("bgraToTensorHalf matches bgraToTensor")
{
	const uint8_t px[16] = {0, 128, 255, 255, 10, 20, 30, 255, 255, 255, 255, 255, 1, 254, 77, 255};
	for (int layout = 0; layout < 2; ++layout) {
		PreprocessOptions o;
		o.layout = layout == 0 ? TensorLayout::NCHW : TensorLayout::NHWC;
		o.order = ChannelOrder::RGB;
		o.mean[0] = 0.485f;
		o.mean[1] = 0.456f;
		o.mean[2] = 0.406f;
		o.std[0] = 0.229f;
		o.std[1] = 0.224f;
		o.std[2] = 0.225f;
		float ref[12];
		uint16_t half[12];
		bgraToTensor(px, 2, 2, 8, o, ref);
		bgraToTensorHalf(px, 2, 2, 8, o, half);
		for (int i = 0; i < 12; ++i) {
			CHECK(half[i] == floatToHalf(ref[i]));
			CHECK(halfToFloat(half[i]) == doctest::Approx(ref[i]).epsilon(1e-3));
		}
	}
}

TEST_CASE("motion estimate: identical frames score 0, a small shift scores above the skip threshold")
{
	const uint32_t w = 640, h = 352, tw = 64, th = 36;
	// Textured frame: soft vertical bars plus a bright square ("a head").
	auto render = [&](int shiftX, int noise, uint32_t seed) {
		std::vector<uint8_t> f(size_t(w) * h * 4);
		for (uint32_t y = 0; y < h; ++y)
			for (uint32_t x = 0; x < w; ++x) {
				const int sx = int(x) - shiftX;
				int v = 60 + int(40.0 * std::sin(sx * 0.05));
				if (sx > 250 && sx < 390 && y > 80 && y < 260)
					v = 200;
				if (noise) {
					seed = seed * 1664525u + 1013904223u;
					v += int(seed >> 28) % (2 * noise + 1) - noise;
				}
				v = std::clamp(v, 0, 255);
				uint8_t *p = &f[(size_t(y) * w + x) * 4];
				p[0] = p[1] = p[2] = uint8_t(v);
				p[3] = 255;
			}
		return f;
	};
	std::vector<uint8_t> a(tw * th), b(tw * th);
	const auto f0 = render(0, 0, 1);
	lumaThumbnail(f0.data(), w, h, w * 4, tw, th, a.data());
	lumaThumbnail(f0.data(), w, h, w * 4, tw, th, b.data());
	CHECK(meanAbsDiff(a.data(), b.data(), a.size()) == 0.0);
	CHECK(a[size_t(18) * tw + 32] >= 195); // the square, averaged
	// Sensor-like noise (+-3 levels per pixel) stays well below the default 1.5.
	const auto n1 = render(0, 3, 7), n2 = render(0, 3, 99);
	lumaThumbnail(n1.data(), w, h, w * 4, tw, th, a.data());
	lumaThumbnail(n2.data(), w, h, w * 4, tw, th, b.data());
	const double noiseDiff = meanAbsDiff(a.data(), b.data(), a.size());
	CHECK(noiseDiff < 1.0);
	// The subject moving by 6 px (~1% of the width) is clearly above it.
	const auto f1 = render(6, 0, 1);
	lumaThumbnail(f0.data(), w, h, w * 4, tw, th, a.data());
	lumaThumbnail(f1.data(), w, h, w * 4, tw, th, b.data());
	const double moveDiff = meanAbsDiff(a.data(), b.data(), a.size());
	MESSAGE("noise " << noiseDiff << ", 6 px shift " << moveDiff);
	CHECK(moveDiff > 1.5);
}
