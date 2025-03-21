import asyncio
import base64
import os
import shlex
import shutil
from enum import StrEnum
from pathlib import Path
from typing import Literal, TypedDict
from uuid import uuid4
import pyautogui
import PIL

from anthropic.types.beta import BetaToolComputerUse20241022Param

from .base import BaseAnthropicTool, ToolError, ToolResult
from .run import run

OUTPUT_DIR = "/tmp/outputs"

TYPING_DELAY_MS = 12
TYPING_GROUP_SIZE = 50

Action = Literal[
    "key",
    "type",
    "mouse_move",
    "left_click",
    "left_click_drag",
    "right_click",
    "middle_click",
    "double_click",
    "screenshot",
    "cursor_position",
]


class Resolution(TypedDict):
    width: int
    height: int


# sizes above XGA/WXGA are not recommended (see README.md)
# scale down to one of these targets if ComputerTool._scaling_enabled is set
MAX_SCALING_TARGETS: dict[str, Resolution] = {
    "XGA": Resolution(width=1024, height=768),  # 4:3
    "WXGA": Resolution(width=1280, height=800),  # 16:10
    "FWXGA": Resolution(width=1366, height=768),  # ~16:9
}


class ScalingSource(StrEnum):
    COMPUTER = "computer"
    API = "api"


class ComputerToolOptions(TypedDict):
    display_height_px: int
    display_width_px: int
    display_number: int | None


def chunks(s: str, chunk_size: int) -> list[str]:
    return [s[i : i + chunk_size] for i in range(0, len(s), chunk_size)]


class ComputerTool(BaseAnthropicTool):
    """
    A tool that allows the agent to interact with the screen, keyboard, and mouse of the current computer.
    The tool parameters are defined by Anthropic and are not editable.
    """

    name: Literal["computer"] = "computer"
    api_type: Literal["computer_20241022"] = "computer_20241022"
    width: int
    height: int
    display_num: int | None

    _screenshot_delay = 0.5
    _scaling_enabled = True

    @property
    def options(self) -> ComputerToolOptions:
        width, height = self.scale_coordinates(
            ScalingSource.COMPUTER, self.width, self.height
        )
        return {
            "display_width_px": width,
            "display_height_px": height,
            "display_number": self.display_num,
        }

    def to_params(self) -> BetaToolComputerUse20241022Param:
        return {"name": self.name, "type": self.api_type, **self.options}

    def __init__(self):
        super().__init__()

        self.width = int(os.getenv("WIDTH") or 0)
        self.height = int(os.getenv("HEIGHT") or 0)
        assert self.width and self.height, "WIDTH, HEIGHT must be set"
        if (display_num := os.getenv("DISPLAY_NUM")) is not None:
            self.display_num = int(display_num)
            self._display_prefix = f"DISPLAY=:{self.display_num} "
        else:
            self.display_num = None
            self._display_prefix = ""

        self.xdotool = f"{self._display_prefix}xdotool"

    async def __call__(
        self,
        *,
        action: Action,
        text: str | None = None,
        coordinate: tuple[int, int] | None = None,
        **kwargs,
    ):
        print("__CALL__ action happening!")
        print(action)
        if action in ("mouse_move", "left_click_drag"):
            if coordinate is None:
                raise ToolError(f"coordinate is required for {action}")
            if text is not None:
                raise ToolError(f"text is not accepted for {action}")
            if not isinstance(coordinate, list) or len(coordinate) != 2:
                raise ToolError(f"{coordinate} must be a tuple of length 2")
            if not all(isinstance(i, int) and i >= 0 for i in coordinate):
                raise ToolError(f"{coordinate} must be a tuple of non-negative ints")

            x, y = self.scale_coordinates(
                ScalingSource.API, coordinate[0], coordinate[1]
            )

            if action == "mouse_move":
                pyautogui.moveTo(x, y)
                screenshot_base64 = (await self.screenshot_lib()).base64_image
                return ToolResult(output="", error="", base64_image=screenshot_base64)
            elif action == "left_click_drag":
                pyautogui.dragTo(x, y, button='left')
                screenshot_base64 = (await self.screenshot_lib()).base64_image
                return ToolResult(output="", error="", base64_image=screenshot_base64)

        if action in ("key", "type"):
            if text is None:
                raise ToolError(f"text is required for {action}")
            if coordinate is not None:
                raise ToolError(f"coordinate is not accepted for {action}")
            if not isinstance(text, str):
                raise ToolError(output=f"{text} must be a string")

            if action == "key":
                pyautogui.press(text)
                screenshot_base64 = (await self.screenshot_lib()).base64_image
                return ToolResult(output="", error="", base64_image=screenshot_base64)
            elif action == "type":
                pyautogui.write(text, interval=TYPING_DELAY_MS/1000)  # Convert ms to seconds
                screenshot_base64 = (await self.screenshot_lib()).base64_image
                return ToolResult(output="", error="", base64_image=screenshot_base64)

        if action in (
            "left_click",
            "right_click",
            "double_click",
            "middle_click",
            "screenshot",
            "cursor_position",
        ):
            if text is not None:
                raise ToolError(f"text is not accepted for {action}")
            if coordinate is not None:
                raise ToolError(f"coordinate is not accepted for {action}")

            if action == "screenshot":
                print("action is screenshot")
                return await self.screenshot_lib()
            elif action == "cursor_position":
                x, y = pyautogui.position()
                x, y = self.scale_coordinates(
                    ScalingSource.COMPUTER,
                    x,
                    y,
                )
                return ToolResult(output=f"X={x},Y={y}", error="", base64_image=None)
            else:
                if action == "left_click":
                    pyautogui.click(button='left')
                elif action == "right_click":
                    pyautogui.click(button='right')
                elif action == "middle_click":
                    pyautogui.click(button='middle')
                elif action == "double_click":
                    pyautogui.doubleClick()
                
                screenshot_base64 = (await self.screenshot_lib()).base64_image
                return ToolResult(output="", error="", base64_image=screenshot_base64)

        raise ToolError(f"Invalid action: {action}")


    async def screenshot_lib(self):
        """Take a screenshot of the current screen and return the base64 encoded image."""
        try:
            output_dir = Path(OUTPUT_DIR)
            output_dir.mkdir(parents=True, exist_ok=True)
            path = output_dir / f"screenshot_{uuid4().hex}.png"

            # Take screenshot using pyautogui
            screenshot = pyautogui.screenshot()
            screenshot.save(path)
            print("screenshot worked so far, going to scale")
            print(self._scaling_enabled)
            if self._scaling_enabled:
                print("do scale")
                x, y = self.scale_coordinates(
                    ScalingSource.COMPUTER, self.width, self.height
                )
                # Resize using PIL instead of convert command
                screenshot = screenshot.resize((x, y), PIL.Image.Resampling.LANCZOS)
            
            # Save the potentially resized image
            screenshot.save(path)

            if path.exists():
                print("screenshot path exists! probably worked")
                return ToolResult(
                    output="",
                    error="",
                    base64_image=base64.b64encode(path.read_bytes()).decode()
                )
            raise ToolError("Failed to save screenshot")
        except ImportError:
            raise ToolError("pyautogui is not installed. Please install with: pip install pyautogui")
        except Exception as e:
            raise ToolError(f"Failed to take screenshot: {str(e)}")
    
    
    async def screenshot_bash(self):
        # this is the old one, preserved in case we need it
        """Take a screenshot of the current screen and return the base64 encoded image."""
        output_dir = Path(OUTPUT_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"screenshot_{uuid4().hex}.png"

        # Try gnome-screenshot first
        if shutil.which("gnome-screenshot"):
            screenshot_cmd = f"{self._display_prefix}gnome-screenshot -f {path} -p"
        else:
            # Fall back to scrot if gnome-screenshot isn't available
            screenshot_cmd = f"{self._display_prefix}scrot -p {path}"

        result = await self.shell(screenshot_cmd, take_screenshot=False)
        if self._scaling_enabled:
            x, y = self.scale_coordinates(
                ScalingSource.COMPUTER, self.width, self.height
            )
            await self.shell(
                f"convert {path} -resize {x}x{y}! {path}", take_screenshot=False
            )

        if path.exists():
            return result.replace(
                base64_image=base64.b64encode(path.read_bytes()).decode()
            )
        raise ToolError(f"Failed to take screenshot: {result.error}")

    async def shell(self, command: str, take_screenshot=True) -> ToolResult:
        """Run a shell command and return the output, error, and optionally a screenshot."""
        _, stdout, stderr = await run(command)
        base64_image = None

        if take_screenshot:
            # delay to let things settle before taking a screenshot
            await asyncio.sleep(self._screenshot_delay)
            base64_image = (await self.screenshot_lib()).base64_image

        return ToolResult(output=stdout, error=stderr, base64_image=base64_image)

    def scale_coordinates(self, source: ScalingSource, x: int, y: int):
        """Scale coordinates to a target maximum resolution."""
        print(f"Scaling coordinates - Source: {source}, Input: ({x}, {y})")
        print(f"Current dimensions: {self.width}x{self.height}, ratio: {self.width/self.height:.3f}")
        
        if not self._scaling_enabled:
            print("Scaling disabled, returning original coordinates")
            return x, y
            
        ratio = self.width / self.height
        target_dimension = None
        for name, dimension in MAX_SCALING_TARGETS.items():
            dim_ratio = dimension["width"] / dimension["height"]
            print(f"Checking {name}: {dimension['width']}x{dimension['height']}, ratio: {dim_ratio:.3f}")
            if abs(dim_ratio - ratio) < 0.1: # 0.02:
                if dimension["width"] < self.width:
                    target_dimension = dimension
                    print(f"Found matching target: {name}")
                    break
                else:
                    print(f"Skipping {name} - width not smaller than current")
            else:
                print(f"Skipping {name} - ratio difference too large: {abs(dim_ratio - ratio):.3f}")
                
        if target_dimension is None:
            print("No suitable target dimension found, returning original coordinates")
            return x, y
            
        x_scaling_factor = target_dimension["width"] / self.width
        y_scaling_factor = target_dimension["height"] / self.height
        print(f"Scaling factors - X: {x_scaling_factor:.3f}, Y: {y_scaling_factor:.3f}")

        if source == ScalingSource.API:
            if x > self.width or y > self.height:
                print(f"Coordinates out of bounds: ({x}, {y})")
                raise ToolError(f"Coordinates {x}, {y} are out of bounds")
            # scale up
            result = (round(x / x_scaling_factor), round(y / y_scaling_factor))
        else:
            # scale down
            result = (round(x * x_scaling_factor), round(y * y_scaling_factor))
            
        print(f"Final coordinates: {result}")
        return result
