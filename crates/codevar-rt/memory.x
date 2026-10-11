/*
 * Default memory map for Codevar's bare-metal `arm` images.
 *
 * These are the numbers of QEMU's `lm3s6965evb` (a Cortex-M3 with 256 KiB of
 * flash at 0 and 64 KiB of SRAM at 0x20000000), which is what
 * `.cargo/config.toml` runs the target binaries under. Any board that differs
 * provides its own `memory.x` on the linker's `-L` path — `link.x` includes
 * whichever `memory.x` comes first, so nothing else has to change.
 */

MEMORY
{
    FLASH (rx)  : ORIGIN = 0x00000000, LENGTH = 256K
    RAM   (rw)  : ORIGIN = 0x20000000, LENGTH = 64K
}

/* Size of the main stack reserved by the `.stack` section in `link.x`. */
STACK_SIZE = 0x4000;
