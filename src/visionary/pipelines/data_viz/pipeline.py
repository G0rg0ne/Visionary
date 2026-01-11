"""
This is a boilerplate pipeline 'data_viz'
generated using Kedro 1.1.1
"""

from kedro.pipeline import Node, Pipeline  # noqa

from .nodes import (
    plot_ticket_evolution,
)

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        Node(
            plot_ticket_evolution,
            inputs="merged_data",
            outputs="ticket_evolution",
            name="plot_ticket_evolution"
        ),
    ])
