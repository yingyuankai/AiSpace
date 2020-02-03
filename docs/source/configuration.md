# Configuration

We use yaml to manage various configurations, and inheritance and override can be implemented between configurations
The configurations of specific task inherit base configuration directly or indirectly.
## Base
```text
configs/base.yml
```
This is the most basic configuration file, including the default configuration about training, logging, etc.

You can read this configuration carefully to understand the possibility of configurability.

## Pretrain

```text
configs/pretrain
```
This kind of configuration file adds pretrained item compared to other configurations mainly, and includes ***base.yml***.

## Specific task

For example:
```text
configs/glue_zh/tnews.yml
```