---
name: process-lifecycle
description: Starting, waiting for, signalling and stopping a long-running program without losing track of it.
---

# Long-running programs

A program that does not return is not a failed command. It is a process you now own.

## Starting

Start it in the background and capture the pid the shell hands back:

    python3 bin/<program>.py &
    pid=$!

Never read a pid back out of `ps`, and never assume one. A pid you did not capture belongs to some other
process, and a signal sent to it either does nothing or hits the wrong thing — both of which look like success
from the outside.

## Waiting

Wait for the readiness the program announces. If it writes a file when it is ready, loop until that file is
there:

    while [ ! -f <ready-file> ]; do sleep 0.1; done

A `sleep` long enough today is a race tomorrow, and a `sleep` shorter than the warm-up means every signal you
send lands before the program is listening.

## Signalling and stopping

Send one signal at a time and confirm each took effect before sending the next. Stop what you started, and
check it is gone.

## Finishing

Read the log the program wrote and confirm the steps appear in the order you meant them to happen. If a step is
missing, you are not finished — the commands returning without an error is not the same as the work being done.
