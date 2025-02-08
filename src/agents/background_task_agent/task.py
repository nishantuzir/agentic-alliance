from typing import Literal  # Import Literal type from typing module
from uuid import uuid4    # Import uuid4 function for generating unique IDs

from langchain_core.messages import BaseMessage  # Import BaseMessage class
from langchain_core.runnables import RunnableConfig  # Import RunnableConfig class

from agents.utils import CustomData  # Import CustomData utility class
from schema.task_data import TaskData  # Import TaskData schema class


class Task:
    def __init__(self, task_name: str) -> None:
        self.name = task_name  # Initialize task name
        self.id = str(uuid4())  # Generate unique ID as string
        self.state: Literal["new", "running", "complete"] = "new"  # Set initial state to "new"
        self.result: Literal["success", "error"] | None = None  # Initialize result as None

    async def _generate_and_dispatch_message(self, config: RunnableConfig, data: dict):
        """Generate and dispatch a task message."""
        task_data = TaskData(name=self.name, run_id=self.id, state=self.state, data=data)
        if self.result:
            task_data.result = self.result  # Add result to task_data if available
        task_custom_data = CustomData(
            type=self.name,
            data=task_data.model_dump(),
        )
        await task_custom_data.adispatch(config)  # Dispatch custom data asynchronously
        return task_custom_data.to_langchain()  # Convert to BaseMessage and return

    async def start(self, config: RunnableConfig, data: dict = {}) -> BaseMessage:
        """Start the task and send initial message."""
        self.state = "new"  # Set state to "new"
        task_message = await self._generate_and_dispatch_message(config, data)
        return task_message  # Return generated message

    async def write_data(self, config: RunnableConfig, data: dict) -> BaseMessage:
        """Update task data while it's running."""
        if self.state == "complete":  # Check if task is complete
            raise ValueError("Only incomplete tasks can output data.")  # Raise error if so
        self.state = "running"  # Set state to "running"
        task_message = await self._generate_and_dispatch_message(config, data)
        return task_message  # Return updated message

    async def finish(
        self, result: Literal["success", "error"], config: RunnableConfig, data: dict = {}
    ) -> BaseMessage:
        """Finish the task and send final result."""
        self.state = "complete"  # Set state to "complete"
        self.result = result  # Record result
        task_message = await self._generate_and_dispatch_message(config, data)
        return task_message  # Return final message