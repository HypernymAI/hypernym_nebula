# Fully static ctxshield-scorer for Linux (x86-64 + ARM64) — build, verify, install into the plugin, test on
# Alpine / Debian / old Ubuntu / busybox. Needs only Docker with buildx (arm64 runs under emulation on x86 hosts).
#
#   make            build both scorers into out/  (fails unless each one is truly static)
#   make verify     run each scorer bare in every test image (+ busybox: no libc at all)
#   make install    copy into ../../ctxshield/static/ and record sha256 in ../PLATFORMS.json
#   make test       install, then the plugin's suites + an end-to-end MCP call in every image × CPU
#   make test-x86_64 / test-aarch64     one CPU only
#   make test-install   (this host's CPU) the REAL install on clean Alpine: website uv line, Claude Code installer, claude plugin
#                       marketplace add + install (this tree), then hooks / status line / MCP as installed
#   make test-install-x86_64 / test-install-aarch64
#   make reproduce      build twice from scratch (no cache) and require byte-identical binaries
#   make clean
#
# Every image and tool is pinned below; change them here only.
.DELETE_ON_ERROR:
SHELL       := /bin/sh
HERE        := $(abspath .)
NATIVE      := $(abspath ..)
PLUGIN      := $(abspath ../../..)
STATIC_DIR  := $(PLUGIN)/ctxshield/ctxshield/static
OUT         := $(HERE)/out
CPUS        := x86_64 aarch64
PLATFORM_x86_64  := linux/amd64
PLATFORM_aarch64 := linux/arm64
UV_IMAGE    := ghcr.io/astral-sh/uv:0.10.7
TEST_IMAGES := alpine:3.20 debian:12-slim ubuntu:18.04
BARE_IMAGE  := busybox:1.36-musl
INSTALL_IMAGE := alpine:3.20
# test-install runs Claude Code itself, whose installer does not finish under qemu emulation: default to this host's
# own CPU. Override (INSTALL_CPUS="x86_64 aarch64") only where both run natively.
HOST_CPU    := $(shell uname -m | sed -e 's/arm64/aarch64/' -e 's/amd64/x86_64/')
INSTALL_CPUS ?= $(HOST_CPU)
SCORERS     := $(foreach c,$(CPUS),$(OUT)/ctxshield-scorer-linux-$(c))
SOURCES     := $(NATIVE)/CMakeLists.txt $(wildcard $(NATIVE)/include/* $(NATIVE)/src/*.cpp $(NATIVE)/src/*.h \
               $(NATIVE)/scorer/* $(NATIVE)/third_party/utf8proc/*) $(HERE)/Dockerfile.build

.PHONY: all build verify install reproduce test test-install $(addprefix test-,$(CPUS)) $(addprefix test-install-,$(CPUS)) clean

all: build
build: $(SCORERS)

$(OUT)/ctxshield-scorer-linux-%: $(SOURCES)
	docker buildx build --platform $(PLATFORM_$*) --build-arg TARGET_ARCH=$* \
	  -f $(HERE)/Dockerfile.build --output type=local,dest=$(OUT) $(NATIVE)
	@test -x $@ || chmod +x $@

$(OUT)/uv-%:
	@mkdir -p $(OUT)
	id=$$(docker create --platform $(PLATFORM_$*) $(UV_IMAGE)) && docker cp $$id:/uv $@ && docker rm $$id >/dev/null
	chmod +x $@

verify: build
	@for c in $(CPUS); do for img in $(TEST_IMAGES) $(BARE_IMAGE); do \
	  printf '%-8s %-20s ' $$c $$img; \
	  docker run --rm --platform $$( [ $$c = x86_64 ] && echo $(PLATFORM_x86_64) || echo $(PLATFORM_aarch64) ) \
	    -v $(OUT):/out:ro $$img /out/ctxshield-scorer-linux-$$c --version || exit 1; \
	done; done

install: build
	mkdir -p $(STATIC_DIR)
	for c in $(CPUS); do install -m 755 $(OUT)/ctxshield-scorer-linux-$$c $(STATIC_DIR)/; done
	python3 $(HERE)/update_platforms.py $(NATIVE)/PLATFORMS.json $(STATIC_DIR) $$(date -u +%Y%m%d_%H%M%S)

test: $(addprefix test-,$(CPUS)) test-install

test-install: $(addprefix test-install-,$(INSTALL_CPUS))

$(addprefix test-install-,$(CPUS)): test-install-%: install
	@echo "════ install flow · $* · $(INSTALL_IMAGE)"
	docker run --rm --platform $(PLATFORM_$*) -v $(abspath $(PLUGIN)/../..):/src:ro \
	  -v $(HERE)/test_claude_install.sh:/test.sh:ro $(INSTALL_IMAGE) sh /test.sh

$(addprefix test-,$(CPUS)): test-%: install $(OUT)/uv-%    # static pattern: .PHONY targets skip implicit rules
	@for img in $(TEST_IMAGES); do \
	  echo "════ $* · $$img"; \
	  docker run --rm --platform $(PLATFORM_$*) -v $(PLUGIN):/src:ro -v $(OUT):/tools:ro \
	    -v $(HERE)/test_in_container.sh:/test.sh:ro $$img sh /test.sh $* || exit 1; \
	done

reproduce:
	@for c in $(CPUS); do for n in 1 2; do \
	  docker buildx build -q --no-cache --platform $$( [ $$c = x86_64 ] && echo $(PLATFORM_x86_64) || echo $(PLATFORM_aarch64) ) \
	    --build-arg TARGET_ARCH=$$c -f $(HERE)/Dockerfile.build --output type=local,dest=$(OUT)/repro$$n $(NATIVE) >/dev/null || exit 1; \
	done; \
	  a=$$(shasum -a 256 $(OUT)/repro1/ctxshield-scorer-linux-$$c | cut -c1-64); b=$$(shasum -a 256 $(OUT)/repro2/ctxshield-scorer-linux-$$c | cut -c1-64); \
	  s=$$(shasum -a 256 $(OUT)/ctxshield-scorer-linux-$$c 2>/dev/null | cut -c1-64); \
	  echo "$$c  build1 $$a  build2 $$b  out/ $$s"; [ "$$a" = "$$b" ] || { echo "NOT REPRODUCIBLE: $$c" >&2; exit 1; }; \
	done; rm -rf $(OUT)/repro1 $(OUT)/repro2; echo reproducible

clean:
	rm -rf $(OUT)
