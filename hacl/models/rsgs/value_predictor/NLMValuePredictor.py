#! /usr/bin/env python3
# -*- coding: utf-8 -*-

from torch import nn

from hacl.nn.nlm import LogicMachine
from .ValuePredictorBase import ValuePredictor


class NLMValuePredictor(ValuePredictor):
    def __init__(
        self,
        n_states,
        input_dims,
        output_dims,
        breadth=None,
        depth=1,
        logic_hidden_dim=[],
        exclude_self=True,
        residual=True,
    ):
        super().__init__(n_states)
        if not isinstance(input_dims, tuple):
            assert breadth is not None
            self.input_dims = tuple([input_dims for i in range(breadth + 1)])
        else:
            self.input_dims = input_dims
        self.output_dims = output_dims
        assert breadth is None or breadth == len(self.input_dims) - 1
        breadth = len(self.input_dims) - 1
        assert breadth >= 0
        self.logic_machine = LogicMachine(
            depth=depth,
            breadth=breadth,
            input_dims=input_dims,
            output_dims=output_dims,
            logic_hidden_dim=logic_hidden_dim,
            exclude_self=exclude_self,
            residual=residual,
        )
        self.decoder = nn.Linear(self.logic_machine.output_dims[1], 1)

    def get_values(self, inputs, *args, **kwargs):
        return self(inputs, *args, **kwargs)

    def forward(self, inputs, *args, **kwargs):
        for k, input in enumerate(inputs):
            assert (self.input_dims[k] == 0 and input is None) or input.size(-1) == self.input_dims[k]
        outputs = self.logic_machine(inputs)
        output = outputs[1]
        assert len(output.size()) == 3 and output.size(1) == self.n_states
        return self.decoder(output.view(-1, output.size(2))).view(output.size(0), output.size(1))
