#!/bin/sh
# Build pypryon's pinned pryon_test oracle (build/oracle/pryon_test) and record
# its digest.
#
# Exits non-zero on any mismatch of the pinned recipe (see the Makefile header):
# the NDK is not r21e (Pkg.Revision 21.4.7075529), a stock link library is
# missing, or the compile/link step fails. A binary sha256 different from PR
# #171's harness digest is reported as a documented divergence, not a failure —
# the recipe is the contract, the digest is corroboration.
#
# ANDROID_NDK_ROOT: Android NDK r21e. PYPRYON_ORACLE_ROOTFS: an unpacked Echo
# rootfs carrying system/lib/libpryon.so (read only).
set -eu

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
: "${ANDROID_NDK_ROOT:?set ANDROID_NDK_ROOT to Android NDK r21e}"
: "${PYPRYON_ORACLE_ROOTFS:?set PYPRYON_ORACLE_ROOTFS to an unpacked Echo rootfs}"
NDK=$ANDROID_NDK_ROOT
CROSS=$NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/armv7a-linux-androideabi24-clang
PIN=107e8079ca73ac0c2d30d4cc1bb2cd62d78ffe8202bf64fa97cc643cdea23f84
NDK_REV=21.4.7075529

die() { echo "build.sh: $*" >&2; exit 1; }

[ -x "$CROSS" ] || die "toolchain wrapper not found: $CROSS (check ANDROID_NDK_ROOT)"
rev=$(sed -n 's/^Pkg.Revision = //p' "$NDK/source.properties")
[ "$rev" = "$NDK_REV" ] || die "NDK Pkg.Revision is '${rev:-missing}', pinned recipe needs '$NDK_REV'"

for f in libpryon.so libz.so; do
    [ -f "$PYPRYON_ORACLE_ROOTFS/system/lib/$f" ] || die "stock library not found: $PYPRYON_ORACLE_ROOTFS/system/lib/$f"
done

echo "== building $ROOT/build/oracle/pryon_test"
make -C "$HERE" all ANDROID_NDK_ROOT="$NDK" PYPRYON_ORACLE_ROOTFS="$PYPRYON_ORACLE_ROOTFS"

BIN=$ROOT/build/oracle/pryon_test
[ -x "$BIN" ] || die "build did not produce $BIN"
sha=$(sha256sum "$BIN" | cut -d' ' -f1)
[ -n "$sha" ] || die "could not digest $BIN"
echo "== sha256 $sha  $BIN"
if [ "$sha" = "$PIN" ]; then
    echo "== digest matches PR #171's pinned harness digest"
else
    echo "== NOTE: digest differs from PR #171's pin ($PIN)."
    echo "   Same sources, same pinned recipe (see tools/oracle/Makefile header):"
    echo "   documented divergence, not a failure."
fi
