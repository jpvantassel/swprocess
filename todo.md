# TODO

> Joseph P. Vantassel

## Before next release

-   Make array1d & sensor1d defensive when reading headers.
-   Update swprocess to lastest possible Python version.
-   Statistics workflow.
-   Change `single` to `nostacking`.
-   _delay in ActiveTimeSeries.
-   Add exists check to first cell of masw notebook to be sure files exist.

## Long term

-   Slow down on Michael's machine.
-   Take noise from the end of the record.
-   Change naming convention in Peaks so that two ambient noise records
at the same time from two different arrays do not conflict. An option to
add a prefix or something.
-   Trimming a record (-0.5 to 1) between (0.1 and 0.95) gives ValueError.
-   Make df a figurative rather than literal attribute.
-   Waveform normalization bnz 0.5m.
-   Add test case for from_max from waterplant.
-   Corr shift test failing.
-   Simplify masw factory pattern to allow straightforward one-off implementations.