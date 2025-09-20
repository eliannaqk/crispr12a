from pydantic import BaseModel, Field

class DataSchema(BaseModel):
    sequence_col: str
    train_data_path: str
    val_data_path: str
    label_col: str | None = None
    batch_size: int = Field(8)
    num_workers: int = Field(2)
    # Optional: enable length-bucketed batching (token-based microbatches)
    token_bucketed_batches: bool = Field(False, description="Enable token-based microbatches with bucketing")
    target_tokens_per_microbatch: int = Field(
        12000,
        description="Approximate target sum of tokens per microbatch when token_bucketed_batches is True",
    )
    bucket_size_multiplier: int = Field(
        10,
        description="Sortish sampler chunk size multiplier relative to batch_size",
    )
    # Optional: random 50% reversal augmentation of sequences in collate
    random_reverse: bool = Field(
        False,
        description="With 50% probability reverse sequences in collate (data augmentation)",
    )

class ModelSchema(BaseModel):
    name: str
    path: str | None = None

    def model_post_init(self, __context):
        assert self.name in ["esm2", "progen2"], self.name
        if self.path is None:
            if self.name == "esm2":
                self.path = "esm2_t33_650M_UR50D"
            elif self.name == "progen2":
                self.path = "base"


class LoRASchema(BaseModel):
    r: int = Field(8, description="LoRA rank")
    alpha: float = Field(16, description="LoRA scaling; typically 2 × r")
    target_modules: list[str] = Field(..., description="Module names to apply LoRA to")
    dropout: float = Field(0.05, description="LoRA dropout probability")
    enabled: bool = Field(True, description="Enable LoRA training")


class AlgorithmsSchema(BaseModel):
    lora: LoRASchema | None = None


class TrainingSchema(BaseModel):
    learning_rate: float = Field(1e-4)
    weight_decay: float | None = Field(None, description="Defaults to learning_rate * 1e-2")
    warmup_steps: int = Field(1000)
    train_steps: int = Field(10000)
    total_lr_decay_factor: float = Field(0.2)
    gradient_clipping_threshold: float = Field(1.0)
    # No explicit gradient accumulation fields; emulate via YAML scaling if needed


class FinetuneAPI(BaseModel):
    save_folder: str
    data: DataSchema
    model: ModelSchema
    training: TrainingSchema
    algorithms: AlgorithmsSchema | None = None
    save_interval_steps: int = Field(1000)
    autoresume: bool = Field(False)
    # Optional: evaluation cadence as raw time string (e.g., "1ep", "300ba")
    eval_interval: str | None = Field(
        default=None,
        description="Evaluation cadence as Composer time string, e.g., '1ep' or '300ba'",
    )
    # Optional: evaluation cadence in batches; if None, trainer default is used
    eval_interval_steps: int | None = Field(
        default=None,
        description="Run evaluation every N batches when set; e.g., 300",
    )
