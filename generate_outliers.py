import numpy as np

from config import args


def perturb_point(point, level, offset=None):
    x, y = int(point // map_size[1]), int(point % map_size[1])
    if offset is None:
        offset = [[0, 1], [1, 0], [-1, 0], [0, -1], [1, 1], [-1, -1], [-1, 1], [1, -1]]
        x_offset, y_offset = offset[np.random.randint(0, len(offset))]
    else:
        x_offset, y_offset = offset
    if 0 <= x + x_offset * level < map_size[0] and 0 <= y + y_offset * level < map_size[1]:
        x += x_offset * level
        y += y_offset * level
    return int(x * map_size[1] + y)


def perturb_batch(batch_x, level, prob):
    noisy_batch_x = []
    for traj in batch_x:
        noisy_batch_x.append(
            [traj[0]] + [perturb_point(p, level) if not p == 0 and np.random.random() < prob else p for p in
                         traj[1: -1]] + [traj[-1]])
    return noisy_batch_x


def generate_outliers(trajs, ratio=args.ratio, level=args.level, point_prob=args.point_prob,
                      observe=args.obeserved_ratio):
    traj_num = len(trajs)
    selected_idx = np.random.randint(0, traj_num, size=int(traj_num * ratio))
    outliers = perturb_batch([trajs[idx] for idx in selected_idx],
                             level=level, prob=point_prob)

    for i, idx in enumerate(selected_idx):
        trajs[idx] = outliers[i]

    new_trajs = []
    for traj in trajs:
        new_trajs.append(traj[: int(len(traj) * observe) + 1])
    return new_trajs, selected_idx


if __name__ == '__main__':
    np.random.seed(args.seed)
    print('============================================================')
    print(f'd = {args.level}' + ', ' + chr(945) + f' = {args.point_prob}' + ', ' + chr(
        961) + f' = {args.obeserved_ratio}')
    print('Dataset: ' + args.dataset)
    map_size = (51, 158)

    data = np.load(f'./data/{args.dataset}/test_data.npy', allow_pickle=True)
    outliers_trajs, outliers_idx = generate_outliers(data)
    outliers_trajs = np.array(outliers_trajs, dtype=object)
    outliers_idx = np.array(outliers_idx)

    np.save(f'./data/{args.dataset}/outliers_data.npy', outliers_trajs)
    np.save(f'./data/{args.dataset}/outliers_idx.npy', outliers_idx)
