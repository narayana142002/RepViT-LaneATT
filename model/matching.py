import torch


def match_proposals_with_targets(head, anchors, targets, t_pos=15.0, t_neg=20.0, min_overlap=2):
    """
    Match anchors to GT lanes using the full lane shape rather than only the
    encoded start point. This is much more tolerant to whether a lane enters
    from the bottom edge or the left/right border.
    """
    anc_xs = anchors[:, 5:]
    tgt_xs = targets[:, 5:]

    overlap = (anc_xs.unsqueeze(1) > 0) & (tgt_xs.unsqueeze(0) > 0)
    overlap_count = overlap.sum(dim=2)

    pair_dist = (anc_xs.unsqueeze(1) - tgt_xs.unsqueeze(0)).abs()
    mean_dist = (pair_dist * overlap).sum(dim=2) / overlap_count.clamp(min=1)

    # Keep a coarse start-point fallback only for tie-breaking when overlap is poor.
    anc_x = anchors[:, 3].unsqueeze(1) * head.img_w
    anc_y = anchors[:, 2].unsqueeze(1) * head.img_h
    tgt_x = targets[:, 3].unsqueeze(0) * head.img_w
    tgt_y = targets[:, 2].unsqueeze(0) * head.img_h
    start_dist = (anc_x - tgt_x).abs() + (anc_y - tgt_y).abs()

    lane_dist = mean_dist.clone()
    lane_dist[overlap_count < min_overlap] = 1e6 + start_dist[overlap_count < min_overlap]

    tgt_idx = lane_dist.argmin(dim=1)
    row_idx = torch.arange(len(anchors), device=anchors.device)
    best_mean = mean_dist[row_idx, tgt_idx]
    best_overlap = overlap_count[row_idx, tgt_idx]

    pos_mask = (best_overlap >= min_overlap) & (best_mean < t_pos)
    neg_mask = (best_overlap < min_overlap) | (best_mean > t_neg)

    return pos_mask, neg_mask, tgt_idx
