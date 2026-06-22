from dataclasses import dataclass

from timecast._internal.logging import setup_logging
from timecast.pipelines._retail import load_retail_dataset
from timecast.schemas import (
    EntryClassicProcess,
    EntrySeasonAnalyticPipeline,
    validate_with_pydantic,
)
from timecast.season.process import ClassicProcess

log = setup_logging()


@dataclass
class SeasonAnalyticPipeline:
    entry: EntrySeasonAnalyticPipeline

    def __post_init__(self):
        pass

    def analyze(self):

        self.classic_dataset, _, dictmerge = load_retail_dataset(self.entry.Dataset)

        proccess = self.entry.Process

        self.classic_process = validate_with_pydantic(EntryClassicProcess)(ClassicProcess)(
            entry={
                "dictmerge": dictmerge,
                "dictdecompose": proccess.DictDecompose,
                "remove_bound": proccess.RemoveBound,
                "plots": proccess.Plots,
                "save_plots": proccess.SavePlots,
                "save_path_plots": proccess.SavePathPlots
            }
        )
        
        self.classic_process.process()
