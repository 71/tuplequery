# tuplequery
##### (Name subject to change)

This crate provides generic utilities for building queries similar to
[DataScript](https://github.com/tonsky/datascript), with a focus on efficiency.

**Warning**: this crate is not sufficiently tested and relies on a fair amount
of `unsafe` code. Its API is also subject to change at any time. I plan on
adding more tests and using it in an actual project at some point, but in the
meantime it probably shouldn't be used for anything important.

A simple example is provided in
[`examples/datascript.rs`](examples/datascript.rs).
