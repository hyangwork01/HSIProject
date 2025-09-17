# 1 加载配置、管理物体动作、记录物体的init位置、

from envs.hoi.components.scenelib.base import BaseSceneLib

class SceneLib(BaseSceneLib):
    def __init__(self,cfg,num_envs: int,device: str = "cpu"):
        super().__init__(cfg,num_envs,device)
        self.create_scene_cfg()

    def create_scene_cfg(self):
        # TODO: 将获取到的物体信息存储到对应的列表里
        self.
    