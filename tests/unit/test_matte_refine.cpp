#include <doctest.h>

#include <vector>

#include "postprocessing/matte_refine.h"

using namespace promatte;

namespace {
std::vector<float> makeMatte(uint32_t w, uint32_t h, float value)
{
	return std::vector<float>(size_t(w) * h, value);
}
} // namespace

TEST_CASE("confidence curve maps threshold band smoothly")
{
	std::vector<float> a = {0.0f, 0.2f, 0.4f, 0.5f, 0.6f, 0.8f, 1.0f};
	MatteRefiner::applyConfidenceCurve(a.data(), a.size(), 0.5f, 0.2f);
	CHECK(a[0] == doctest::Approx(0.f));
	CHECK(a[1] == doctest::Approx(0.f));
	CHECK(a[2] > 0.1f);
	CHECK(a[2] < 0.5f);
	CHECK(a[3] == doctest::Approx(0.5f));
	CHECK(a[4] > 0.5f);
	CHECK(a[4] < 0.9f);
	CHECK(a[5] == doctest::Approx(1.f));
	CHECK(a[6] == doctest::Approx(1.f));
}

TEST_CASE("confidence curve with zero softness is a hard threshold")
{
	std::vector<float> a = {0.49f, 0.5f, 0.51f};
	MatteRefiner::applyConfidenceCurve(a.data(), a.size(), 0.5f, 0.0f);
	CHECK(a[0] == 0.f);
	CHECK(a[1] == 1.f);
	CHECK(a[2] == 1.f);
}

TEST_CASE("erode shrinks and dilate grows a square")
{
	const uint32_t w = 16, h = 16;
	auto a = makeMatte(w, h, 0.f);
	for (uint32_t y = 4; y < 12; ++y)
		for (uint32_t x = 4; x < 12; ++x)
			a[y * w + x] = 1.f;
	auto count = [&](const std::vector<float> &m) {
		int c = 0;
		for (float v : m)
			c += v > 0.5f;
		return c;
	};
	CHECK(count(a) == 64);
	MatteRefiner r;
	auto eroded = a;
	r.morphShift(eroded.data(), w, h, -1);
	CHECK(count(eroded) == 36); // 6x6
	auto dilated = a;
	r.morphShift(dilated.data(), w, h, 1);
	CHECK(count(dilated) == 100); // 10x10
	auto dilated2 = a;
	r.morphShift(dilated2.data(), w, h, 2);
	CHECK(count(dilated2) == 144); // 12x12
}

TEST_CASE("component filter removes small blobs but keeps the person and fills holes")
{
	const uint32_t w = 64, h = 64;
	auto a = makeMatte(w, h, 0.f);
	// person: 30x40 block
	for (uint32_t y = 10; y < 50; ++y)
		for (uint32_t x = 15; x < 45; ++x)
			a[y * w + x] = 1.f;
	// hole inside the person 3x3
	for (uint32_t y = 25; y < 28; ++y)
		for (uint32_t x = 25; x < 28; ++x)
			a[y * w + x] = 0.f;
	// stray blob 3x3 in the corner
	for (uint32_t y = 2; y < 5; ++y)
		for (uint32_t x = 55; x < 58; ++x)
			a[y * w + x] = 1.f;
	MatteRefiner r;
	int removed = r.filterComponents(a.data(), w, h, 0.01f, 0.01f);
	CHECK(removed == 1);
	CHECK(a[3 * w + 56] == 0.f);  // blob removed
	CHECK(a[26 * w + 26] == 1.f); // hole filled
	CHECK(a[30 * w + 30] == 1.f); // person intact
	CHECK(a[0] == 0.f);           // background intact
}

TEST_CASE("full refine pipeline is idempotent on a clean binary matte")
{
	const uint32_t w = 32, h = 32;
	auto a = makeMatte(w, h, 0.f);
	for (uint32_t y = 8; y < 24; ++y)
		for (uint32_t x = 8; x < 24; ++x)
			a[y * w + x] = 1.f;
	MatteRefiner r;
	RefineParams p;
	p.minComponentArea = 0.005f;
	auto before = a;
	r.process(a.data(), w, h, p);
	CHECK(a == before);
}

TEST_CASE("matting models: only the tails are cut, the mid-range keeps its alpha")
{
	std::vector<float> a = {0.0f, 0.02f, 0.2f, 0.5f, 0.8f, 0.98f, 1.0f};
	MatteRefiner::applyMatteTails(a.data(), a.size(), 0.04f, 0.96f);
	CHECK(a[0] == 0.f);
	CHECK(a[1] == 0.f);
	CHECK(a[2] == doctest::Approx(0.2f).epsilon(0.1)); // (0.2-0.04)/0.92 = 0.174
	CHECK(a[3] == doctest::Approx(0.5f));
	CHECK(a[4] == doctest::Approx(0.8f).epsilon(0.05));
	CHECK(a[5] == 1.f);
	CHECK(a[6] == 1.f);
}

TEST_CASE("process: matting models keep hair alpha, segmentation models keep the threshold curve")
{
	const uint32_t w = 8, h = 8;
	RefineParams p;
	p.threshold = 0.5f;
	p.softness = 0.02f + 0.43f * 0.35f; // default smoothness
	MatteRefiner r;

	auto seg = makeMatte(w, h, 0.2f);
	r.process(seg.data(), w, h, p);
	CHECK(seg[0] == doctest::Approx(0.f)); // below the smoothstep band: cut

	p.alphaMatting = true;
	auto mat = makeMatte(w, h, 0.2f);
	r.process(mat.data(), w, h, p);
	CHECK(mat[0] > 0.15f);
	CHECK(mat[0] < 0.21f);
	auto lo = makeMatte(w, h, 0.02f);
	r.process(lo.data(), w, h, p);
	CHECK(lo[0] == 0.f);
	auto hi = makeMatte(w, h, 0.98f);
	r.process(hi.data(), w, h, p);
	CHECK(hi[0] == 1.f);
}

TEST_CASE("fill holes: matting models leave holes the model rates as background")
{
	const uint32_t w = 64, h = 64;
	auto base = makeMatte(w, h, 0.f);
	for (uint32_t y = 10; y < 50; ++y)
		for (uint32_t x = 15; x < 45; ++x)
			base[y * w + x] = 1.f;
	// gap between the fingers: 3x3, model alpha 0.05
	for (uint32_t y = 25; y < 28; ++y)
		for (uint32_t x = 25; x < 28; ++x)
			base[y * w + x] = 0.05f;
	// uncertain miss inside the body: 3x3, model alpha 0.4
	for (uint32_t y = 35; y < 38; ++y)
		for (uint32_t x = 25; x < 28; ++x)
			base[y * w + x] = 0.4f;
	MatteRefiner r;

	auto seg = base;
	r.filterComponents(seg.data(), w, h, 0.f, 0.01f);
	CHECK(seg[26 * w + 26] == 1.f);
	CHECK(seg[36 * w + 26] == 1.f);

	auto mat = base;
	r.filterComponents(mat.data(), w, h, 0.f, 0.01f, kMattingHoleKeepAlpha);
	CHECK(mat[26 * w + 26] == doctest::Approx(0.05f)); // real gap: untouched
	CHECK(mat[36 * w + 26] == 1.f);                    // model miss: filled

	// Same through process() with the matting flag.
	RefineParams p;
	p.maxHoleArea = 0.01f;
	p.alphaMatting = true;
	auto full = base;
	r.process(full.data(), w, h, p);
	CHECK(full[26 * w + 26] < 0.02f); // stays background: (0.05 - 0.04) / 0.92
	CHECK(full[36 * w + 26] == 1.f);
}
