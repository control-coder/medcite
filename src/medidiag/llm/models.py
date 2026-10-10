"""应用使用的 MiMo 模型名。

``mimo-v2.5`` 在 2026-10-14 下线，此后只能回放已有的调用记录；新的真实调用一律用 ``mimo-v2.6-flash``。
两个名字都允许出现在请求里：旧名字用于回放旧调用记录，新名字用于真实调用与新的调用记录。
"""

PREVIOUS_MIMO_MODEL = "mimo-v2.5"
ACTIVE_MIMO_MODEL = "mimo-v2.6-flash"
ALLOWED_MIMO_MODELS = (PREVIOUS_MIMO_MODEL, ACTIVE_MIMO_MODEL)
