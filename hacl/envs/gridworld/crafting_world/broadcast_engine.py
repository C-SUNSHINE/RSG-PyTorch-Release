#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from jactorch.nn import TorchApplyRecorderMixin

from hacl.envs.gridworld.crafting_world.configs import OBJECT2IDX, TYPE2IDX
from hacl.envs.gridworld.crafting_world.v20210515 import CraftingWorldV20210515
from hacl.envs.gridworld.crafting_world.engine.engine import CraftingWorldActions, DIRECTION_VEC

Actions = CraftingWorldActions

OBJECT_DIM = 55
STATUS_DIM = 2
INVENTORY_DIM = 9
DEFAULT_BLOCK_DIM = 10
TYPE_DIM = 6
# STATE_DIM = 6 + INVENTORY_DIM + BLOCK_DIM * 5


class CraftingWorldBroadcastEngine(TorchApplyRecorderMixin):
    def __init__(self, env_args=None, implement='broadcast', pai=False, object_limit=None):
        super().__init__()
        if env_args is not None:
            self.env_args = CraftingWorldV20210515.complete_env_args(env_args)
            self.dataset = self.env_args['dataset'] if 'dataset' in env_args else None
            self.env = CraftingWorldV20210515(env_args=env_args)
            self.object_limit = DEFAULT_BLOCK_DIM if 'object_limit' not in self.env_args else self.env_args['object_limit']
        self.implement = implement
        self.pai = pai
        if object_limit is not None:
            self.object_limit = object_limit
        self.state_dim = 6 + INVENTORY_DIM + self.object_limit * 5
        print('CraftingWorldBroadcastEngine, implement=%s, object_limit=%d' % (self.implement, self.object_limit))

    def _state2tensor(self, state):
        global_states = list(state[:6])
        inventory_states = list(state[6])
        assert len(inventory_states) == state[5]
        inventory_states.extend([0] * (INVENTORY_DIM - state[5]))
        block_states = []
        for block_state in state[7]:
            block_states.extend(block_state)
        block_states.extend([0, 0, 0, 0, 0] * (self.object_limit - len(state[7])))
        res = torch.LongTensor(global_states + inventory_states + block_states)
        assert res.size(0) == self.state_dim
        return res

    def states2tensor(self, states):
        res = torch.stack([self._state2tensor(state) for state in states], dim=0)
        return res

    def _tensor2state(self, x):
        assert len(x.size()) == 1 and x.size(0) == self.state_dim
        global_states = x[:6].tolist()
        inventory_states = tuple(x[6:6 + INVENTORY_DIM].tolist()[:global_states[5]])
        block_states = []
        block_tensor = x[6 + INVENTORY_DIM:].view(self.object_limit, 5)
        for i in range(block_tensor.size(0)):
            if block_tensor[i, 0] != 0:
                block_states.append(tuple(block_tensor[i].tolist()))
        block_states = tuple(block_states)
        return tuple(global_states + [inventory_states, block_states])

    def tensor2states(self, x):
        states = [self._tensor2state(x[i]) for i in range(x.size(0))]
        return states

    def _comperss_size(self, x):
        if len(x.size()) == 1:
            siz = None
        else:
            siz = x.size()[:-1]
        return x.view(-1, x.size(-1)), siz

    def _restore_size(self, x, siz):
        if siz is None:
            return x.view(*x.size()[1:])
        return x.view(*siz, *x.size()[1:])

    def _action_move(self, x, d):
        n = x.size(0)
        new_xy = x[:, :2] + torch.LongTensor(DIRECTION_VEC[d]).to(x.device).view(1, 2)
        s_inventory = x[:, 6:6 + INVENTORY_DIM].reshape(n, INVENTORY_DIM)
        s_blocks = x[:, 6 + INVENTORY_DIM:].reshape(n, self.object_limit, 5)
        mask_blocks = ~s_blocks[:, :, 0].eq(0)
        xy_blocks = s_blocks[:, :, 2:4]
        overlap = (xy_blocks.eq(new_xy.unsqueeze(1)).min(2)[0] & mask_blocks) & s_blocks[:, :, 1].eq(TYPE2IDX['structure'])
        has_key = s_inventory.eq(OBJECT2IDX['key']).max(1)[0]
        has_boat = s_inventory.eq(OBJECT2IDX['boat']).max(1)[0]
        is_switch = s_blocks[:, :, 0].eq(OBJECT2IDX['switch'])
        is_door = s_blocks[:, :, 0].eq(OBJECT2IDX['door'])
        is_open = s_blocks[:, :, 4].eq(1)
        is_water = s_blocks[:, :, 0].eq(OBJECT2IDX['water'])
        walkable = is_switch | (is_door & (has_key.unsqueeze(1) | is_open)) | (is_water & has_boat.unsqueeze(1))
        collide = (overlap & ~walkable).max(1)[0]
        outside = new_xy[:, 0].lt(0) | new_xy[:, 0].ge(x[:, 2]) | new_xy[:, 1].lt(0) | new_xy[:, 1].ge(x[:, 2])
        valid = ~(outside | collide)
        mixed_xy = x[:, :2].clone()
        mixed_xy[valid] = new_xy[valid]
        new = torch.cat([mixed_xy, x[:, 2:]], dim=1)
        return new, valid

    def _agent_full(self, s):
        return s[:, 5].eq(s[:, 4])

    def _agent_holding(self, ivt, item):
        item_id = OBJECT2IDX[item]
        return ivt.eq(item_id).max(1)[0]

    def _agent_push(self, s, ivt, mask, item):
        if isinstance(item, str):
            items = OBJECT2IDX[item]
        else:
            items = item.unsqueeze(1)
        masked_s = s[mask]
        masked_ivt = ivt[mask]
        masked_ivt.scatter_(1, masked_s[:, 5].unsqueeze(1), items)
        masked_s[:, 5] += 1
        s[mask] = masked_s
        ivt[mask] = masked_ivt
        assert s[:, 5].le(s[:, 4]).min()
        return s, ivt

    def _agent_find(self, s, ivt, item):
        item_id = OBJECT2IDX[item]
        ivt_index = torch.arange(INVENTORY_DIM, dtype=torch.long, device=s.device).view(1, -1)
        item_pos = (ivt.eq(item_id) * (ivt_index + 1)).max(1)[0] - 1
        assert item_pos.min() >= 0
        return item_pos

    def _agent_pop(self, s, ivt, mask, item):
        masked_s = s[mask]
        masked_ivt = ivt[mask]
        item_pos = self._agent_find(masked_s, masked_ivt, item)
        item_end = masked_s[:, 5] - 1
        end_item = masked_ivt.gather(1, item_end.unsqueeze(1))
        masked_ivt.scatter_(1, item_pos.unsqueeze(1), end_item)
        masked_ivt.scatter_(1, item_end.unsqueeze(1), 0)
        masked_s[:, 5] -= 1
        s[mask] = masked_s
        ivt[mask] = masked_ivt
        assert s[:, 5].le(s[:, 4]).min()
        return s, ivt

    def _overlap_block(self, s, blc):
        xy = s[:, :2]
        xy_blocks = blc[:, :, 2:4]
        mask_blocks = ~blc[:, :, 0].eq(0)
        overlap = (xy_blocks.eq(xy.unsqueeze(1)).min(2)[0] & mask_blocks)
        return overlap

    def _on_block(self, s, blc, block_type=None, block_name=None):
        overlap = self._overlap_block(s, blc)
        if block_type is not None:
            is_type = blc[:, :, 1].eq(TYPE2IDX[block_type])
            overlap = overlap & is_type
        if block_name is not None:
            is_block = blc[:, :, 0].eq(OBJECT2IDX[block_name])
            overlap = overlap & is_block
        return overlap.max(1)[0]

    def _action_toggle_switch(self, x):  # Agent must be standing on a switch
        n = x.size(0)
        s = x[:, :6]
        ivt = x[:, 6:6 + INVENTORY_DIM]
        blc = x[:, 6 + INVENTORY_DIM:].reshape(-1, self.object_limit, 5)

        is_switch = blc[:, :, 0].eq(OBJECT2IDX['switch'])
        assert is_switch.sum(1).max() <= 1
        switches = blc[is_switch]
        switches[:, 4] = 1 - switches[:, 4]
        blc[is_switch] = switches

        has_switches = is_switch.max(1)[0].view(n, 1).expand(n, blc.size(1))
        switch_status = torch.zeros(n, blc.size(1), dtype=torch.long, device=x.device)
        switch_status[is_switch] = switches[:, 4]
        switch_status = switch_status.max(1)[0].view(n, 1).expand(n, blc.size(1))

        is_door_with_switch = blc[:, :, 0].eq(OBJECT2IDX['door']) & has_switches
        doors = blc[is_door_with_switch]
        doors[:, 4] = switch_status[is_door_with_switch]
        blc[is_door_with_switch] = doors
        return torch.cat([s, ivt.view(n, -1), blc.view(n, -1)], dim=1), torch.ones(n, dtype=torch.bool, device=x.device)

    def _action_toggle_grab(self, x):  # Agent must be standing on a pick-able item
        n = x.size(0)
        s = x[:, :6]
        ivt = x[:, 6:6 + INVENTORY_DIM]
        blc = x[:, 6 + INVENTORY_DIM:].reshape(-1, self.object_limit, 5)

        mask = ~self._agent_full(s)
        overlap = self._overlap_block(s, blc)
        items = blc[overlap]
        s, ivt = self._agent_push(s, ivt, mask, items[mask][:, 0].clone())
        items[mask] = 0
        blc[overlap] = items
        return torch.cat([s, ivt.view(n, -1), blc.view(n, -1)], dim=1), mask

    def _action_toggle_mine(self, x):  # Agent must be standing on a resource
        n = x.size(0)
        s = x[:, :6]
        ivt = x[:, 6:6 + INVENTORY_DIM]
        blc = x[:, 6 + INVENTORY_DIM:].reshape(-1, self.object_limit, 5)
        valid = torch.zeros(n, dtype=torch.bool, device=x.device)
        for rule in self.env.engine.rules:
            if rule['rule_name'].startswith('mine_'):
                mask = self._on_block(s, blc, block_name=rule['location']) & ~self._agent_full(s)
                if mask.max():
                    for tool in rule['holding']:
                        mask &= self._agent_holding(ivt, tool)
                    if mask.max():
                        s, ivt = self._agent_push(s, ivt, mask, rule['create'])
                        valid |= mask
        return torch.cat([s, ivt.view(n, -1), blc.view(n, -1)], dim=1), valid

    def _action_toggle_craft(self, x):  # Agent must be standing on a station
        n = x.size(0)
        s = x[:, :6]
        ivt = x[:, 6:6 + INVENTORY_DIM]
        blc = x[:, 6 + INVENTORY_DIM:].reshape(-1, self.object_limit, 5)
        valid = torch.zeros(n, dtype=torch.bool, device=x.device)
        for rule in self.env.engine.rules:
            if rule['rule_name'].startswith('craft_'):
                mask = self._on_block(s, blc, block_name=rule['location']) & ~valid
                if mask.max():
                    for ingredient in rule['recipe']:
                        mask &= self._agent_holding(ivt, ingredient)
                    if mask.max():
                        for ingredient in rule['recipe']:
                            s, ivt = self._agent_pop(s, ivt, mask, ingredient)
                        s, ivt = self._agent_push(s, ivt, mask, rule['create'])
                        valid |= mask
        return torch.cat([s, ivt.view(n, -1), blc.view(n, -1)], dim=1), valid

    def _action_toggle(self, x):
        n = x.size(0)
        xy = x[:, :2]
        s_blocks = x[:, 6 + INVENTORY_DIM:].reshape(n, self.object_limit, 5)
        on_switch = self._on_block(xy, s_blocks, block_name='switch')
        on_item = self._on_block(xy, s_blocks, block_type='item')
        on_resource = self._on_block(xy, s_blocks, block_type='resource')
        on_station = self._on_block(xy, s_blocks, block_type='station')

        valid = torch.zeros(x.size(0), dtype=torch.bool, device=x.device)
        new = x.clone()

        if on_switch.max():
            new_switch, valid_switch = self._action_toggle_switch(x[on_switch])
            new[on_switch] = new_switch
            valid[on_switch] |= valid_switch
        if on_item.max():
            new_grab, valid_grab = self._action_toggle_grab(x[on_item])
            new[on_item] = new_grab
            valid[on_item] |= valid_grab
        if on_resource.max():
            new_mine, valid_mine = self._action_toggle_mine(x[on_resource])
            new[on_resource] = new_mine
            valid[on_resource] |= valid_mine
        if on_station.max():
            new_craft, valid_craft = self._action_toggle_craft(x[on_station])
            new[on_station] = new_craft
            valid[on_station] |= valid_craft
        return new, valid

    def action_broadcast(self, x, action, inplace=True):
        a = Actions(action)
        if not inplace:
            x = x.clone()

        x, siz = self._comperss_size(x)
        if a in [Actions.Up, Actions.Down, Actions.Left, Actions.Right]:
            new, valid = self._action_move(x, a)
        elif a == Actions.Toggle:
            new, valid = self._action_toggle(x)
        else:
            raise ValueError('Invalid action %d' % action)

        return self._restore_size(new, siz), self._restore_size(valid, siz)

    def action_brute_force(self, x, action, inplace=True):
        if not inplace:
            x = x.clone()
        x, siz = self._comperss_size(x)
        states = self.tensor2states(x)
        new_states = []
        valid = []
        env = CraftingWorldV20210515(self.env_args)
        for state in states:
            env.load_from_symbolic_state(state)
            success = env.action(action)
            new_states.append(env.get_symbolic_state())
            valid.append(success)
        y = self.states2tensor(new_states).to(x.device)
        valid = torch.BoolTensor(valid).to(x.device)
        return self._restore_size(y, siz), self._restore_size(valid, siz)

    def action(self, x, action, inplace=True):
        if self.implement == 'brute_force':
            return self.action_brute_force(x, action, inplace=inplace)
        elif self.implement == 'broadcast':
            if self.pai:
                new, valid = self.action_broadcast(x, action, inplace=inplace)
                new_b, valid_b = self.action_brute_force(x, action, inplace=inplace)
                assert valid_b.eq(valid).min()

                new_states = self.tensor2states(new[valid])
                new_states_b = self.tensor2states(new_b[valid_b])
                env = CraftingWorldV20210515(self.env_args)
                for s, sb in zip(new_states, new_states_b):
                    env.load_from_symbolic_state(s)
                    ps = env.get_symbolic_state()
                    env.load_from_symbolic_state(sb)
                    psb = env.get_symbolic_state()
                    assert ps == psb
            else:
                new, valid = self.action_broadcast(x, action, inplace=inplace)
            return new, valid


if __name__ == '__main__':
    pass