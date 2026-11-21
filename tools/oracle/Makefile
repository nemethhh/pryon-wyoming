# Pinned oracle build: pryon_test (armv7 PIE), pypryon's ground-truth oracle.
#
# The two sources are verbatim copies (plus attribution headers) of
# github.com/Gamer92000/echo-dot-assist src/tools/pryon_test.c and
# src/include/pryon_api.h, the harness whose reproducible build upstream
# PR #171 (vendored under vendor/pryon-native) pins:
#
#   Android NDK r21e   (Pkg.Revision 21.4.7075529, Clang 9.0.9)
#   CC                 armv7a-linux-androideabi24-clang
#   CFLAGS             -O2 -Wall -Wextra -fPIE -I<pryon_api.h dir>
#   LDFLAGS            -pie -fuse-ld=lld -Wl,--allow-shlib-undefined \
#                       -Wl,--unresolved-symbols=ignore-in-shared-libs
#   link libs          <rootfs>/system/lib/{libpryon.so,libz.so}
#
# PR #171's harness digest:
#   107e8079ca73ac0c2d30d4cc1bb2cd62d78ffe8202bf64fa97cc643cdea23f84
# A different sha256 with this exact recipe and these exact sources is a
# documented divergence (build.sh prints the note), never a build failure —
# the recipe is the contract, the digest is corroboration.
#
# ANDROID_NDK_ROOT: Android NDK r21e. PYPRYON_ORACLE_ROOTFS: an unpacked Echo
# rootfs (read only). Output goes to <repo>/build/oracle/ (gitignored).

HERE    := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))
ROOT    := $(abspath $(HERE)/../..)
NDK     := $(ANDROID_NDK_ROOT)
CROSS   := $(NDK)/toolchains/llvm/prebuilt/linux-x86_64/bin/armv7a-linux-androideabi24-clang
STOCK   := $(PYPRYON_ORACLE_ROOTFS)/system/lib
OUT     := $(ROOT)/build/oracle

CFLAGS  := -O2 -Wall -Wextra -fPIE -I$(HERE)
LDFLAGS := -pie -fuse-ld=lld -Wl,--allow-shlib-undefined -Wl,--unresolved-symbols=ignore-in-shared-libs

.PHONY: all clean
all: $(OUT)/pryon_test

$(OUT)/pryon_test: pryon_test.c pryon_api.h $(STOCK)/libpryon.so $(STOCK)/libz.so
	@mkdir -p $(OUT)
	$(CROSS) $(CFLAGS) pryon_test.c -o $@ $(LDFLAGS) $(STOCK)/libpryon.so $(STOCK)/libz.so

clean:
	rm -rf $(OUT)
