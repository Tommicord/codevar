/*
 * Default memory map for Codevar's bare-metal `arm` images.
 *
 * These are the numbers of QEMU's `mps2-an385` (a Cortex-M3 with 4 MiB of
 * code memory at 0 and 4 MiB of SRAM at 0x20000000), which is what
 * `.cargo/config.toml` runs the target binaries under. Any board that
 * differs provides its own `memory.x` on the linker's `-L` path — `link.x`
 * includes whichever `memory.x` comes first, so nothing else has to change.
 */

MEMORY
{
    FLASH (rx)  : ORIGIN = 0x00000000, LENGTH = 4M
    RAM   (rw)  : ORIGIN = 0x20000000, LENGTH = 4M
}

/* Size of the main stack reserved by the `.stack` section in `link.x`. */
STACK_SIZE = 0x4000;
