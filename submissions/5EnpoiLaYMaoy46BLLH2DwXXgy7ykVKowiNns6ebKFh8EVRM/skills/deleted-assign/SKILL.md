---
name: deleted-assign
description: OPEN on NameError, UnboundLocalError, or a missing field. The assignment was deleted or moved. Scan every name in the function, not only the one in the traceback.
---

# Restoring a deleted or moved assignment

The traceback or the issue quotes a name. That name is the first site, not the only site. Work in this order and edit as soon as the first site is clear.

## 1. The quoted name

`search_files` the name across the package, then read only the function the traceback points at.

- Every read of the name, and every assignment. A read that runs on a path where no assignment has happened yet is the site.
- An assignment that sits **after** a `return` does not count. Read the lines that follow each `return` in the function.
- An assignment that was moved **below** its first use is the same bug from the other side. Move it back above the use. Do not invent a new value if a later assignment already has the right one.
- A parameter overwritten before it is read, or a name that exists only in another branch, is the same family.

## 2. The rest of that function

After the first restore, read the function from top to bottom once. For each local name, ask whether every path that reads it also assigns it first. These injections delete several assignments in one function, and each one you put back is separate credit. Stop only when that pass finds nothing.

Also look for absence, which has no error yet:

- A branch whose body is `pass`, `...`, or a comment, next to a sibling branch that builds a real object.
- A constructor or a call with fewer keyword arguments than the same call elsewhere in the file.
- A class method that its siblings all define and this class does not. List the methods. The missing one is the site.
- Output that is missing a field, a column, or a line. Go to the function that **builds** that object, not the function that prints it, and compare it with the sibling builder.

## 3. Where the missing text comes from

You are putting lines back, so you need a source for what they said. In order:

- The other branch of the same condition, when the two arms should build the same kind of value.
- A sibling function in the same file that does the same job for another format, dialect, or grant.
- The docstring and the type hints of the function you are in.
- An assertion in a surviving test: the expected string, the expected attribute, the expected `None`.

Use that evidence. Do not design a new implementation.

## 4. One site, then prove it

Apply one restoration. Re-run `/tmp/repro.py`. If the quoted name is gone and a new name is now undefined, that is the next deleted assignment — restore it the same way. If the repro is clean, run the single covering test file before you hunt further, so a bad second edit cannot throw away the first.

Leave the function when a name is merely unusual and every read is still assigned. Unusual is not evidence.
