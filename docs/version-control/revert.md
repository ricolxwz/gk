---
title: 版本控制:回退
comments: true
---

Git回退版本主要有两个命令: 一个是`git reset`, 一个是`git revert`. `get reset`用于回退版本, 可以遗弃不再使用的提交. `git revert`在当前提交的后面, 新增一次提交, 抵消掉上次提交导致的所有变化, 不会改变过去的历史, 主要用于安全地取消过去发布的提交.

## `git reset`

`git reset`有三种模式, 分别为`--mixed`, `--hard`和`--soft`.

### `--mixed`

=== "`--mixed`实验1"

    当前`git log`: 

    ```bash
    $ git log --oneline
    4f22c66 (HEAD -> master) add "test this file"
    25869ac add "hello world"
    d90a575 create testfile.txt
    ```

    当前`git status`:

    ```bash
    $ git status
    位于分支 master
    尚未暂存以备提交的变更：
    （使用 "git add <文件>..." 更新要提交的内容）
    （使用 "git restore <文件>..." 丢弃工作区的改动）
            修改：     testfile.txt

    修改尚未加入提交（使用 "git add" 和/或 "git commit -a"）
    ```

    我们在`4f22c65`提交之后, 又加了一行`Something new...`, 并且还没有将其加入到暂存区里面.

    ![](https://img.ricolxwz.io/ca8f91f8fc04b7c531daf5f3bfba93ac.png)
    
    现在`git reset --mixed 25869ac`.

    执行后的`git log`:

    ```bash
    $ git log --oneline
    25869ac (HEAD -> master) add "hello world"
    d90a575 create testfile.txt
    ```

    执行后的`git status`:

    ```bash
    $ git status
    位于分支 master
    尚未暂存以备提交的变更：
    （使用 "git add <文件>..." 更新要提交的内容）
    （使用 "git restore <文件>..." 丢弃工作区的改动）
            修改：     testfile.txt

    修改尚未加入提交（使用 "git add" 和/或 "git commit -a"）
    ```

    `Test this file.`和`Something new...`这两行都在工作区中, 暂存区空.

    ![](https://img.ricolxwz.io/4448a7cf70abf4c08fb9d723cd14d43e.png)

=== "`--mixed`实验2"

    当前`git log`:

    ```bash
    $ git log --oneline
    4ef41a4 (HEAD -> master) add "test this file"
    731166d add "hello world"
    1c67d2a create testfile.txt
    ```

    当前`git status`:

    ```bash
    $ git status
    位于分支 master
    要提交的变更：
    （使用 "git restore --staged <文件>..." 以取消暂存）
            修改：     testfile.txt
    ```

    我们在`4ef41a4`提交之后, 又加了一行`Something new...`, 并且已经将其加入到暂存区里面.

    ![](https://img.ricolxwz.io/10b76838fb3bbf7c99c62eeefe7e753c.png)

    现在执行`git reset --mixed 731166d`.

    执行后的`git log`:

    ```bash
    $ git log --oneline
    731166d (HEAD -> master) add "hello world"
    1c67d2a create testfile.txt
    ```

    执行后的`git status`:

    ```bash
    $ git status
    位于分支 master
    尚未暂存以备提交的变更：
    （使用 "git add <文件>..." 更新要提交的内容）
    （使用 "git restore <文件>..." 丢弃工作区的改动）
            修改：     testfile.txt

    修改尚未加入提交（使用 "git add" 和/或 "git commit -a"）
    ```

    `Test this file.`和`Something new...`这两行都在工作区中, 暂存区空.

    ![](https://img.ricolxwz.io/4448a7cf70abf4c08fb9d723cd14d43e.png)

### `--soft`

=== "`--soft`实验1"

    当前`git log`:

    ```bash
    $ git log --oneline
    0762fbd (HEAD -> master) add "test this file"
    4e65054 add "hello world"
    57331cc create testfile.txt
    ```

    当前`git status`:

    ```bash
    $ git status
    位于分支 master
    尚未暂存以备提交的变更：
    （使用 "git add <文件>..." 更新要提交的内容）
    （使用 "git restore <文件>..." 丢弃工作区的改动）
            修改：     testfile.txt

    修改尚未加入提交（使用 "git add" 和/或 "git commit -a"）
    ```

    我们在`0762fbd`提交之后, 又加了一行`Something new...`, 并且还没有将其加入到暂存区里面.

    ![](https://img.ricolxwz.io/ca8f91f8fc04b7c531daf5f3bfba93ac.png)

    现在`git reset --soft 4e65054`.

    执行后的`git log`:

    ```bash
    $ git log --oneline
    4e65054 (HEAD -> master) add "hello world"
    57331cc create testfile.txt
    ```

    执行后的`git status`:

    ```bash
    $ git log --oneline
    4e65054 (HEAD -> master) add "hello world"
    57331cc create testfile.txt
    (test) [wenzexu@archlinux test3]$ git status
    位于分支 master
    要提交的变更：
    （使用 "git restore --staged <文件>..." 以取消暂存）
            修改：     testfile.txt

    尚未暂存以备提交的变更：
    （使用 "git add <文件>..." 更新要提交的内容）
    （使用 "git restore <文件>..." 丢弃工作区的改动）
            修改：     testfile.txt
    ```

    `Test this file.`这一行在暂存区中, `Something new...`这一行在工作区中

    ![](https://img.ricolxwz.io/ca8f91f8fc04b7c531daf5f3bfba93ac.png)

=== "`--soft`实验2"

    当前`git log`:

    ```bash
    $ git log --oneline
    74cb841 (HEAD -> master) add "test this file"
    95dcb90 add "hello world"
    79a455e create testfile.txt
    ```

    当前`git status`:

    ```bash
    $ git status
    位于分支 master
    要提交的变更：
    （使用 "git restore --staged <文件>..." 以取消暂存）
            修改：     testfile.txt
    ```

    我们在`74cb841`提交之后, 又加了一行`Something new...`, 并且已经将其加入到暂存区里面.

    ![](https://img.ricolxwz.io/10b76838fb3bbf7c99c62eeefe7e753c.png)

    现在`git reset --soft 95dcb90`.

    执行后的`git log`:

    ```bash
    $ git log --oneline
    95dcb90 (HEAD -> master) add "hello world"
    79a455e create testfile.txt
    ```

    执行后的`git status`:

    ```bash
    位于分支 master
    要提交的变更：
    （使用 "git restore --staged <文件>..." 以取消暂存）
            修改：     testfile.txt
    ```

    `Test this file.`和`Something new...`这两行都在暂存区中.

    ![](https://img.ricolxwz.io/10b76838fb3bbf7c99c62eeefe7e753c.png)

### `--hard`

=== "`--hard`实验1"

    当前`git log`:

    ```bash
    $ git log --oneline
    77203bc (HEAD -> master) add "test this file"
    1361b5b add "hello world"
    c7286db create testfile.txt
    ```

    当前`git status`:

    ```bash
    $ git status
    位于分支 master
    尚未暂存以备提交的变更：
    （使用 "git add <文件>..." 更新要提交的内容）
    （使用 "git restore <文件>..." 丢弃工作区的改动）
            修改：     testfile.txt

    修改尚未加入提交（使用 "git add" 和/或 "git commit -a"）
    ```

    我们在`77203bc`提交之后, 又加了一行`Something new...`, 并且还没有将其加入到暂存区里面.

    ![](https://img.ricolxwz.io/ca8f91f8fc04b7c531daf5f3bfba93ac.png)

    现在`git reset --hard 1361b5b`.

    执行后的`git log`:

    ```bash
    $ git log --oneline
    1361b5b (HEAD -> master) add "hello world"
    c7286db create testfile.txt
    ```

    执行后的`git status`:

    ```bash
    $ git status
    位于分支 master
    无文件要提交，干净的工作区
    ```

    `Test this file.`和`Something new...`这两行都没了. 暂存区空.

    ![](https://img.ricolxwz.io/cf5ce09023b22928e2c54d6dbff93594.png)

=== "`--hard`实验2"

    当前`git log`:

    ```bash
    $ git log --oneline
    ec190e2 (HEAD -> master) add "test this file"
    e49eef4 add "hello world"
    acf1962 create testfile.txt
    ```

    当前`git status`:

    ```bash
    $ git status
    位于分支 master
    尚未暂存以备提交的变更：
    （使用 "git add <文件>..." 更新要提交的内容）
    （使用 "git restore <文件>..." 丢弃工作区的改动）
            修改：     testfile.txt

    修改尚未加入提交（使用 "git add" 和/或 "git commit -a"）
    ```

    我们在`ec190e2`提交之后, 又加了一行`Something new...`, 并且已经将其加入到暂存区里面.

    ![](https://img.ricolxwz.io/10b76838fb3bbf7c99c62eeefe7e753c.png)

    现在`git reset --hard e49eef4`.

    执行后的`git log`:

    ```bash
    $ git log --oneline
    e49eef4 (HEAD -> master) add "hello world"
    acf1962 create testfile.txt
    ```

    执行后的`git status`:

    ```bash
    $ git status
    位于分支 master
    无文件要提交，干净的工作区
    ```

    `Test this file.`和`Something new...`这两行都没了. 暂存区空.

    ![](https://img.ricolxwz.io/cf5ce09023b22928e2c54d6dbff93594.png)

### 总结

- `--mixed`: 清空暂存区, 工作区原有的内容也不变, 将暂存区原来的内容以及原节点和目标节点之间的差异放到工作区中
- `--soft`: 工作区原有的内容不变, 暂存区原有的内容也不变, 将原节点和目标节点之间的所有差异放到暂存区中
- `--hard`: 清空暂存区和工作区, 不保留任何差异