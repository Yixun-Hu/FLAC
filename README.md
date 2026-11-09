# exp_38 E0 artifacts (not code)

The sealed E0 PASS receipt and the eight E0 cells (metrics / stream / receipt JSONs) produced on the A6000 box
(mae-cab-lab-server) on 2026-11-08 at the exp_38 pin df84993724666bc4d972a8c177b5e9ada7bfbecd
(closure 31142c792b76e29d43b715c7bd955adf0e184292543ea3b7987bafaafabf6694). Every neuronic FULL job re-admits them.

Extract into the neuronic campaign worktree (only these paths; they stay untracked there):

    git -C <campaign tree> fetch origin exp-38-e0-artifacts
    git -C <campaign tree> archive FETCH_HEAD worklog outputs_FLAC | tar -x -C <campaign tree>
    cd <campaign tree> && git show FETCH_HEAD:E0_ARTIFACTS.sha256 | sha256sum -c

This branch has no common history with the code branches and is never merged.
