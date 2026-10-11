/*
 * Codevar bare-metal link script (Cortex-M).
 *
 * A hosted toolchain gets its memory map, entry point and startup code from
 * the C runtime; a Cortex-M core gets all three from the image itself. This
 * script therefore does three jobs that the default `rust-lld` script does
 * not:
 *
 *   1. pins `__vector_table` at the start of `FLASH`, so the core reads the
 *      initial stack pointer and `Reset` from addresses 0 and 4;
 *   2. defines `ENTRY(Reset)` and the `_stack_top`/`.data`/`.bss` symbols the
 *      reset stub copies and clears before the binary's `_start` runs;
 *   3. lays `.data` out in `RAM` with a flash load image (`AT > FLASH`),
 *      which is what makes that copy possible.
 *
 * The `MEMORY` block and `STACK_SIZE` come from `memory.x`, found through the
 * `-L` path the binary's `build.rs` adds. Point that at your own directory
 * with a different `memory.x` to retarget the board.
 */

INCLUDE memory.x

ENTRY(Reset)

SECTIONS
{
    /* Word 0 is the initial MSP, word 1 the reset address: keep it first
       and keep everything before it out of the image. */
    .vector_table :
    {
        KEEP(*(.vector_table))
    } > FLASH

    .text :
    {
        . = ALIGN(4);
        *(.text .text.* .gnu.linkonce.t.*)
        *(.glue_7 .glue_7t .vfp11_16 .vfp4_16)
        KEEP(*(.init))
        KEEP(*(.fini))
        . = ALIGN(4);
    } > FLASH

    .rodata :
    {
        . = ALIGN(4);
        *(.rodata .rodata.* .gnu.linkonce.r.*)
        . = ALIGN(4);
    } > FLASH

    .ARM.extab :
    {
        *(.ARM.extab* .gnu.linkonce.armextab.*)
    } > FLASH

    .ARM.exidx :
    {
        __exidx_start = .;
        *(.ARM.exidx* .gnu.linkonce.armexidx.*)
        __exidx_end = .;
    } > FLASH

    /* Load address of `.data`; `codevar_rt_reset` copies it to `_sdata`. */
    _sidata = LOADADDR(.data);

    .data :
    {
        . = ALIGN(4);
        _sdata = .;
        *(.data .data.* .gnu.linkonce.d.*)
        . = ALIGN(4);
        _edata = .;
    } > RAM AT > FLASH

    .bss (NOLOAD) :
    {
        . = ALIGN(4);
        _sbss = .;
        *(.bss .bss.* .gnu.linkonce.b.*)
        *(COMMON)
        . = ALIGN(4);
        _ebss = .;
    } > RAM

    /* The main stack lives in the image so a board with no startup code
       still has a valid MSP before `Reset` runs. */
    .stack (NOLOAD) :
    {
        . = ALIGN(8);
        _stack_bottom = .;
        . += STACK_SIZE;
        . = ALIGN(8);
        _stack_top = .;
    } > RAM

    /DISCARD/ :
    {
        *(.comment)
        *(.note.GNU-stack)
        *(.note.gnu.build-id)
    }
}
