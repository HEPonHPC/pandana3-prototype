# Planning notes


## Purpose

This document is intended to be a very fluid record of the short-term plans for development of PandAna3.
Items that are found to have long-term value should be migrated to another document.

## Testing

### History

It seems to be difficult to come to a right level of detail for functional testing.
The first round of testing created tests that were too large.
The result was tests which were becoming hard to maintain.
It was also hard to understand whether the necessary behaviors were being tested.

The second round of revisions were an attempt to reduce the size of these tests.
This was an attempt at individual tests for each method or function.
This resulted in tests that were too fine-grained.
Again, it was difficult to maintain the tests because of the sheer bulk, and the amount of repetition.
The introduction of the use of fixture helped somewhat to reduce the repetition, but the bulk of tests was still becoming too hard to manage.

The current round of testing is concentrating on:

* testing visible behaviors in the *happy path*
* testing important failure modes

### Testing `Var`s and `Cut`s


### Behavior to test

There are 3 different groups of behaviors that seems relevant:

1. newly-created (before `prepare` is called)
2. *prepared* (after `prepare` is called)
3. evaluation (checking the returned value)

#### Newly created `Var`s

Many of these tests do not require a file.
For those that do, they should work when using a dummy file, because none will actually read the file.

* they are not *prepared*
* they can report what tables will be read
* they can report what the column names in the resulting `pd.DataFrame` will be
* some functions raise exceptions:
    * `eval`
    * `inq_datasets_read`

#### *Prepared* `Var`s

These tests all require a file.
Happy-path tests require a well-formed file.
Failure mode paths may require specifically engineered bad files.

* calling `prepare` transitions to the *prepared* state
* ill-formed `Var`s raise exceptions
* can report datasets to be read, which includes index datasets


#### Evaluation

These tests all require a well-formed file.

* result is a `pd.DataFrame`
* result has expected columns
* columns match those reported by the `Var`.
* correct index columns are present
