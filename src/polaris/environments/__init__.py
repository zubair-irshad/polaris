import os
import gymnasium as gym
from polaris.environments.manager_based_rl_splat_environment import (
    ManagerBasedRLSplatEnv,
)
from polaris.environments.droid_cfg import EnvCfg as DroidCfg
from isaaclab.envs import ManagerBasedRLEnv

# Import rubric system
from polaris.environments.rubrics import Rubric
from polaris.utils import DATA_PATH
import polaris.environments.rubrics.checkers as checkers


# =============================================================================
# Environment Registration
# =============================================================================

gym.register(
    id='DROID-BlockStackKitchen',
    entry_point=ManagerBasedRLSplatEnv,
    kwargs={
        "env_cfg_entry_point": DroidCfg,
        "usd_file": str(DATA_PATH / "block_stack_kitchen/scene.usda"),
        "rubric": Rubric(
            criteria=[
                checkers.reach("green_cube", threshold=0.2),
                checkers.reach("wood_cube", threshold=0.2),
                (checkers.lift("green_cube", default_height=0.06, threshold=0.03), [0]),
                (checkers.lift("wood_cube", default_height=0.06, threshold=0.03), [1]),
                (checkers.is_within_xy("green_cube", "tray", 0.8), [2]),
                (checkers.is_within_xy("wood_cube", "tray", 0.8), [3]),
                (checkers.is_within_xy("green_cube", "wood_cube", 0.5), [4, 5]),
            ]
        ),
    },
    disable_env_checker=True,
    order_enforce=False,
)


gym.register(
    id="DROID-FoodBussing",
    entry_point=ManagerBasedRLSplatEnv,
    disable_env_checker=True,
    order_enforce=False,
    kwargs={
        "env_cfg_entry_point": DroidCfg,
        "usd_file": str(DATA_PATH / "food_bussing/scene.usda"),
        "rubric": Rubric(
            criteria=[
                checkers.reach("ice_cream_", threshold=0.2),
                checkers.reach("grapes", threshold=0.2),
                (checkers.lift("ice_cream_", threshold=0.06), [0]),
                (checkers.lift("grapes", threshold=0.06), [1]),
                (
                    checkers.is_within_xy("ice_cream_", "bowl", percent_threshold=0.8),
                    [2],
                ),
                (checkers.is_within_xy("grapes", "bowl", percent_threshold=0.8), [3]),
            ]
        ),
    },
)

gym.register(
    id="DROID-PanClean",
    entry_point=ManagerBasedRLSplatEnv,
    disable_env_checker=True,
    order_enforce=False,
    kwargs={
        "env_cfg_entry_point": DroidCfg,
        "usd_file": str(DATA_PATH / "pan_clean/scene.usda"),
        "rubric": Rubric(
            criteria=[
                checkers.reach("sponge", threshold=0.2),
                (checkers.lift("sponge", threshold=0.09, default_height=0.0), [0]),
                (checkers.is_within_xy("sponge", "pan", percent_threshold=0.8), [1]),
            ]
        ),
    },
)


gym.register(
    id="DROID-MoveLatteCup",
    entry_point=ManagerBasedRLSplatEnv,
    disable_env_checker=True,
    order_enforce=False,
    kwargs={
        "env_cfg_entry_point": DroidCfg,
        "usd_file": str(DATA_PATH / "move_latte_cup/scene.usda"),
        "rubric": Rubric(
            criteria=[
                checkers.reach("latteartcup_eval", threshold=0.2),
                (checkers.lift("latteartcup_eval", threshold=0.04), [0]),
                (checkers.is_within_xy("latteartcup_eval", "cuttingboard_eval", percent_threshold=0.8), [1]),
            ]
        ),
    },
)

gym.register(
    id="DROID-OrganizeTools",
    entry_point=ManagerBasedRLSplatEnv,
    disable_env_checker=True,
    order_enforce=False,
    kwargs={
        "env_cfg_entry_point": DroidCfg,
        "usd_file": str(DATA_PATH / "organize_tools/scene.usda"),
        "rubric": Rubric(
            criteria=[
                checkers.reach("scissor", threshold=0.2),
                (checkers.lift("scissor", threshold=0.04), [0]),
                (checkers.is_within_xy("scissor", "container_01", percent_threshold=0.8), [1]),
            ]
        ),
    },
)

# =============================================================================
# ManipVerse scenes
# =============================================================================
# Assembled via the manipverse pipeline (stage 07). Asset path is read from
# the MANIPVERSE_SCENE0_USD env var so it stays out of source control. The
# rubric is intentionally empty for now — we register the env primarily for
# rendering / sysid validation while we settle the gsplat 3DGS backend.

# Default to PolaRiS-Hub/droid_manipverse_scene0/scene.usda — same layout
# convention as the other registered scenes. Override with
# MANIPVERSE_SCENE0_USD if the asset lives elsewhere.
_manipverse_scene0 = os.environ.get(
    "MANIPVERSE_SCENE0_USD",
    str(DATA_PATH / "droid_manipverse_scene0/scene.usda"),
)

gym.register(
    id="DROID-ManipVerse-Scene0",
    entry_point=ManagerBasedRLSplatEnv,
    disable_env_checker=True,
    order_enforce=False,
    kwargs={
        "env_cfg_entry_point": DroidCfg,
        "usd_file": _manipverse_scene0,
        "rubric": Rubric(criteria=[]),
    },
)

gym.register(
    id="DROID-ManipVerse-Scene5",
    entry_point=ManagerBasedRLSplatEnv,
    disable_env_checker=True,
    order_enforce=False,
    kwargs={
        "env_cfg_entry_point": DroidCfg,
        "usd_file": str(DATA_PATH / "droid_manipverse_scene5/scene.usda"),
        "rubric": Rubric(
            criteria=[
                checkers.reach("white_paper_plate__0", threshold=0.2),
                (checkers.lift("white_paper_plate__0", threshold=0.04), [0]),
            ]
        ),
    },
)

gym.register(                                                                                                                                                                       
    id="DROID-ManipVerse-Scene13",                                                                                                                                                  
    entry_point=ManagerBasedRLSplatEnv,                                                                                                                                             
    disable_env_checker=True,                                                                                                                                                       
    order_enforce=False,                                                                                                                                                            
    kwargs={                                                                                                                                                                        
        "env_cfg_entry_point": DroidCfg,                                                                                                                                            
        "usd_file": str(DATA_PATH / "droid_manipverse_scene13/scene.usda"),                                                                                                         
        "rubric": Rubric(                                                                                                                                                           
            criteria=[                                                                                                                                                              
                checkers.reach("green_snack_packet", threshold=0.2),                                                                                                                
                (checkers.lift("green_snack_packet", threshold=0.04), [0]),                                                                                                         
            ]                                                                                                                                                                       
        ),                                                                                                                                                                          
    },                                                                                                                                                                              
)        

gym.register(
    id="DROID-ManipVerse-Scene1",
    entry_point=ManagerBasedRLSplatEnv,
    disable_env_checker=True,
    order_enforce=False,
    kwargs={
        "env_cfg_entry_point": DroidCfg,
        "usd_file": str(DATA_PATH / "droid_manipverse_scene1/scene.usda"),
        "rubric": Rubric(
            criteria=[
                checkers.reach("yellow_patterned_mug", threshold=0.2),
                (checkers.lift("yellow_patterned_mug", threshold=0.04), [0]),
            ]
        ),
    },
)


gym.register(
    id="DROID-TapeIntoContainer",
    entry_point=ManagerBasedRLSplatEnv,
    disable_env_checker=True,
    order_enforce=False,
    kwargs={
        "env_cfg_entry_point": DroidCfg,
        "usd_file": str(DATA_PATH / "tape_into_container/scene.usda"),
        "rubric": Rubric(
            criteria=[
                checkers.reach("tape_00", threshold=0.2),
                (checkers.lift("tape_00", threshold=0.04), [0]),
                (checkers.is_within_xy("tape_00", "container_02", percent_threshold=0.8), [1]),
            ]
        ),
    },
)


gym.register(
    id="DROID-ManipVerse-TriData1",
    entry_point=ManagerBasedRLSplatEnv,
    disable_env_checker=True,
    order_enforce=False,
    kwargs={
        "env_cfg_entry_point": DroidCfg,
        "usd_file": str(DATA_PATH / "droid_manipverse_tridata1/scene.usda"),
        "rubric": Rubric(
            criteria=[
                checkers.reach("red_apple", threshold=0.2),
                (checkers.lift("red_apple", threshold=0.04), [0]),
            ]
        ),
    },
)
