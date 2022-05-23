## Code formatting

Source code should follow the formatting conventions established in the project.
There are several software configurations to help developers format their code:

1. [.editorconfig](.editorconfig) specifies basic formatting elements, such as
   character set and indentation width. Many text editors and IDEs provide
   built-in support or a plugin for _editorconfig_, see
   https://editorconfig.org/ for details.
2. The [.clang-format](src/.clang-format) file contains detailed specification
   for C++ source code formatting. Automatic formatting may be applied with
   [clang-format][clang-format] version 13 or later. Use the
   [run-clang-format.py](scripts/run-clang-format.py) script for parallel
   processing and correct formatting of CUDA kernel launches and OpenMP pragmas.
   The script is also run in a continuous integration job on Gitlab to give
   developers hints about what should be reformatted.
3. Source files in the `src/TNL/` directory should include a copyright header at
   the top. The [update-copyright-headers.py](scripts/update-copyright-headers.py)
   script can be used to generate or update copyright headers in source files.

When you are unsure or if something is not specified in the aforementioned
configuration, rely on [mirroring][mirroring] the formatting style of existing
code.

> __Note:__ Code formatting changes should be applied _before making a commit_,
> i.e., all commits should be already formatted correctly. Ideally, there should
> be no separate commits to fix formatting issues. When working on a feature
> branch, you can squash separate commits fixing style issues into previous
> commits which introduced the problem (see below on how to use the interactive
> rebase).

[clang-format]: https://clang.llvm.org/docs/ClangFormat.html
[mirroring]: https://en.wikipedia.org/wiki/Mirroring_(psychology)

## Code linting

The continuous integration on Gitlab features a code linting job using
[clang-tidy](https://clang.llvm.org/extra/clang-tidy/) to help developers
diagnosing and fixing common programming errors. The configuration is provided
in the [.clang-tidy](src/.clang-tidy) file. Various editors and IDEs provide
[integrations](https://clang.llvm.org/extra/clang-tidy/Integrations.html) either
directly for _clang-tidy_ or for [clangd](https://clangd.llvm.org/) (which
integrates _clang-tidy_). This way you can get full linting directly in your
editor without having to rely on the continuous integration.

## How to configure git

It is important to [configure your git username and email address](
https://mmg-gitlab.fjfi.cvut.cz/gitlab/help/gitlab-basics/start-using-git.md#add-your-git-username-and-set-your-email),
since every git commit will use this information to identify you as the author:

    git config --global user.name "John Doe"
    git config --global user.email "john.doe@example.com"

In the TNL project, this username should be your real name (given name and
family name) and the email address should be the email address used in the
Gitlab profile. You should use the same configuration on all computers where
you make commits. If you have made some commits with a different email address
in the past, you can also add secondary email addresses to the Gitlab profile.

## How to write good commit messages

Begin with a short summary line a.k.a. message subject:

- Use up to 50 characters; this is the git official preference.
- Finish without a sentence-ending period.

Continue with a longer description a.k.a. message body:

- Add a blank line after the summary line, then write as much as you want.
- Use up to 72 characters per line for typical text for word wrap.
- Use as many characters as needed for atypical text, such as URLs, terminal
  output, formatted messages, etc.
- Include any kind of notes, links, examples, etc. as you want.

See [5 Useful Tips For A Better Commit Message](
https://robots.thoughtbot.com/5-useful-tips-for-a-better-commit-message) and
[How to Write a Git Commit Message](https://chris.beams.io/posts/git-commit/)
for examples and reasoning.

## How to split commits

Extensive changes should be split into multiple commits so that only related
changes are made in each commit. All changes made in a commit should be
described in its commit message. If describing all changes would not result in
a good commit message, you should probably make multiple separate commits.

Multiple small commits are better than one big commit, because later they can be
easily [squashed](https://git-scm.com/book/en/v2/Git-Tools-Rewriting-History)
together, whereas splitting a big commit into logical parts is significantly
more difficult.

> __Tip:__ Use `git add -p` to stage only part of your working tree for the next
> commit. See [git add -p: The most powerful git feature you're not using yet](
https://johnkary.net/blog/git-add-p-the-most-powerful-git-feature-youre-not-using-yet/).

## Rebase-based workflow

The development of new features should follow the rebase-based workflow:

- Create a new _feature_ branch based on the main branch (`develop`).
- Make commits with your work in the feature branch.
- When there are other new commits in the `develop` branch, you should do a
  [rebase](https://git-scm.com/book/en/v2/Git-Branching-Rebasing) to rewind your
  commits on top of the current develop branch.
    - If there are conflicts, you will need to resolve them manually. Hence, it
      is a good practice to rebase as often as possible (generally as soon as
      the changes appear in the `develop` branch) and to split commits into
      logical parts.
- When your work is ready for review, you can open a [merge request](
  https://mmg-gitlab.fjfi.cvut.cz/gitlab/tnl/tnl-dev/merge_requests/new).
    - If the branch is not ready for merge yet, prepend `[WIP]` to the merge
      request title to indicate _work in progress_ and to prevent a premature
      merge.
    - This is also a good time to squash small commits (e.g. typos, forgotten
      changes or trivial corrections) with relevant bigger commits to make the
      review easier.
- When your work in the _feature_ branch is finished, you should also do an
  [interactive rebase](https://thoughtbot.com/blog/git-interactive-rebase-squash-amend-rewriting-history)
  and squash small commits (notably when fixing typos or compilation problems),
  reword commit messages (check all typos) and overall make the history _nice_.
- When reviewed, the feature branch can be merged into the develop branch.

The main advantages of this workflow are linear history, clear commits and
reduction of merge conflicts. See [A rebase-based workflow](
https://brokenco.de/2010/04/02/a-rebase-based-workflow.html) and
[Why is rebase-then-merge better than just merge](https://stackoverflow.com/a/457988)
([complement](https://stackoverflow.com/a/804178)) for reference.
