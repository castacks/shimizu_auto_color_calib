import cv2
import numpy as np
import time
import matplotlib
if True:
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

AREA_THRESHOLD = 0.1
STD_THRESHOLD = 0.1
EXPOSURE_THRESHOLD = 0.005
CRSTD_THRESHOLD = 0.16
MIN_PATCH_RATIO_THRESHOLD = 1 / (40 * 2)
MIN_INTERVAL_THRESHOLD = 1 / (16 * 2)
LINE_MIN_LENGTH = 4
FIT_K = 3
FIT_LAMBDA = 0.1
FIT_INTERVAL = 5
COEFF_LENGTH = 100

ENABLE_PLT = True
ENABLE_EXPIRED_HINT = True


x_base, y_base = None, None

fit_fig = plt.figure(figsize=(4, 3), dpi=80)
# fit_img = None


show = np.zeros((1000, 640 * 1 + 640 + 200 + 640, 3), np.float32)
expired_hint = np.zeros((100, 100))
expired_hint[:] = np.arange(0, 100, 1) % 2 * 255
expired_hint = cv2.resize(expired_hint, (show.shape[1], show.shape[0]), interpolation=cv2.INTER_NEAREST)

to_be_list = []
coeff = np.array([1, 1, 1])

last_fit_time = 0
xs, ys = [], []
ATA = None
ATb = None
alpha = None
raw_shape = None
show_line = None

fade_raw_dist = None
fade = None


def make_fade(alpha):
    global fade_raw_dist
    if fade_raw_dist is None:
        x_raw_base, y_raw_base = np.meshgrid(range(raw_shape[1]), range(raw_shape[0]))
        fade_raw_dist = (((x_raw_base - raw_shape[1] / 2))**2 + ((y_raw_base - raw_shape[0] / 2))**2)**0.5 / max(raw_shape)

    mul = np.zeros(raw_shape[:2])
    k = FIT_K
    for i in range(0, k + 1):
        mul += alpha[i] * fade_raw_dist**(i * 2)
    mul /= mul.max()
    return mul


def incremental_fit(x, y, k=FIT_K, lamb=FIT_LAMBDA):
    global ATA, ATb, alpha, xs, ys, last_fit_time, show_line, fade
    xs.append(x)
    ys.append(y)

    if not (time.time() - last_fit_time > FIT_INTERVAL and len(xs) > 0):
        return

    if ATA is None:
        reg = np.identity(FIT_K + 1) * FIT_LAMBDA
        reg[0, 0] = 0
        ATA = reg
        ATb = np.zeros(k + 1)

    last_fit_time = time.time()
    xs = np.array(xs)
    ys = np.array(ys)
    n = len(ys)
    A = np.ones((n, k + 1))
    for i in range(1, k + 1):
        A[:, i] = xs**(i * 2)

    ATA += np.matmul(A.T, A)
    ATb += np.matmul(A.T, ys)

    alpha = np.matmul(np.linalg.inv(ATA), ATb)
    alpha[1:] = np.minimum(alpha[1:], 0)

    xs, ys = [], []

    xt = np.linspace(0, 0.5)
    yt = np.zeros_like(xt)
    for i in range(0, k + 1):
        yt += alpha[i] * xt**(i * 2)

    if show_line is not None:
        show_line.pop(0).remove()
    if ENABLE_PLT:
        show_line = plt.plot(xt, yt)
        plt.ylim(0, 0.5)
        plt.draw()
    fade = make_fade(alpha)


def process(raw, block, point, settings={}):
    """
    raw: (H,W,3) 0-1
    block: (480,640) 0-1
    point: (480,640) 0-1
    =>
    para
    show (?,?,3) 0-1
    """
    for i in settings:
        globals()[i] = settings[i]

    global raw_shape
    if raw_shape is None:
        raw_shape = raw.shape

    show_list = {}
    ret = pickout(raw, block, point, show_list)

    global show
    if ENABLE_EXPIRED_HINT:
        show[..., 0] -= 255 - expired_hint
        show[..., -1] += expired_hint
    raw_show = cv2.resize(raw, (640, 480))
    show[:raw_show.shape[0], :raw_show.shape[1]] = raw_show
    show[:block.shape[0], raw_show.shape[1]:raw_show.shape[1] + block.shape[1]] = cv2.cvtColor(block, cv2.COLOR_GRAY2BGR)
    show[block.shape[0]:block.shape[0] + point.shape[0], raw_show.shape[1]:raw_show.shape[1] + block.shape[1]] = cv2.cvtColor(point, cv2.COLOR_GRAY2BGR)

    global coeff
    if ret is not None:
        new_coeff, intensity = ret
        coeff = new_coeff
        incremental_fit(intensity[0], intensity[1])

    balanced_show = raw_show * coeff

    show[raw_show.shape[0]:raw_show.shape[0] * 2, :raw_show.shape[1]] = balanced_show
    global fade
    if fade is not None:
        if fade.shape != balanced_show.shape:
            fade = cv2.resize(fade, (balanced_show.shape[1], balanced_show.shape[0]))
        show[block.shape[0]:block.shape[0] * 2, -block.shape[1]:] = balanced_show / fade[..., None]

    for i in show_list:
        if len(show_list[i].shape) == 2:
            show_list[i] = cv2.cvtColor(show_list[i], cv2.COLOR_GRAY2BGR)

    try:
        show[block.shape[0]:block.shape[0] + point.shape[0], raw_show.shape[1]:raw_show.shape[1] + block.shape[1]] = show_list["c_show"]
        show[:200, raw_show.shape[1] + block.shape[1]:raw_show.shape[1] + block.shape[1] + 200] = show_list["warp"]
        show[200:200 * 2, raw_show.shape[1] + block.shape[1]:raw_show.shape[1] + block.shape[1] + 200] = show_list["canny"]
        show[200 * 2:200 * 3, raw_show.shape[1] + block.shape[1]:raw_show.shape[1] + block.shape[1] + 200] = show_list["canny_labels"]
        show[200 * 3:200 * 4, raw_show.shape[1] + block.shape[1]:raw_show.shape[1] + block.shape[1] + 200] = show_list["matrix"]
        show[200 * 4:200 * 5, raw_show.shape[1] + block.shape[1]:raw_show.shape[1] + block.shape[1] + 200] = show_list["warp_calibrated"]
        show[240:240 * 2, -block.shape[1]:-block.shape[1] + 320] = cv2.resize(fade, (320, 240))[..., None]
        show[:240, -block.shape[1]:-block.shape[1] + 320] = show_list["fit_img"]
    except KeyError:
        pass

    para = {}
    para["coeff"] = coeff
    para["length"] = len(to_be_list)
    para["alpha"] = alpha
    return para, show


def pickout(raw, rect, c, show_list):

    global x_base, y_base
    if x_base is None:
        x_base, y_base = np.meshgrid(range(c.shape[1]), range(c.shape[0]))

    rect = cv2.dilate(rect, np.ones((10, 10)))
    rect_b = (rect > 0.05).astype(np.uint8)
    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(rect_b)
    bgid = labels[rect_b == 0].min()
    stats[bgid, 4] = -1
    tgtid = np.argmax(stats[:, -1])
    rect[labels != tgtid] = 0
    c *= rect

    c_mask = (c > 0.1).astype(np.uint8) * 255

    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(c_mask)

    ps = []

    c_show = c.copy()

    for i in range(1, retval):
        mask = (labels == i).astype(np.uint8)
        mask = cv2.dilate(mask, np.ones((10, 10)))
        pc = mask * c
        x = (pc * x_base).sum() / pc.sum()
        y = (pc * y_base).sum() / pc.sum()
        ps.append((-pc.sum(), (x, y)))
        cv2.circle(c_show, (int(x), int(y)), 10, 0.2)

    if len(ps) < 4:
        return
    ps = [i[1] for i in sorted(ps)[:4]]
    ps = np.array(ps)
    center = ps.mean(axis=0)
    ps = sorted(ps, key=lambda p: np.arctan2((p - center)[1], (p - center)[0]))
    ps = np.array(ps)

    for i, p in enumerate(ps):
        cv2.circle(c_show, (int(p[0]), int(p[1])), 10, 255)
        cv2.putText(c_show, str(i), (int(p[0]), int(p[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, 255)
    show_list["c_show"] = c_show

    ps[:, 0] = ps[:, 0] / c.shape[1] * raw.shape[1]
    ps[:, 1] = ps[:, 1] / c.shape[0] * raw.shape[0]

    tw, th = 200, 200
    cp = np.array([(0, 0), (tw, 0), (tw, th), (0, th)])

    H, _ = cv2.findHomography(
        ps,
        cp
    )

    warp = cv2.warpPerspective(raw, H, (tw, th))
    show_list["warp"] = warp

    canny = cv2.Canny((warp * 255).astype(np.uint8), 30, 200)
    canny = cv2.dilate(canny, None)
    canny = cv2.erode(canny, None)
    canny = 255 - canny

    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(canny, connectivity=4)
    area = stats[:, -1:]
    widths = stats[:, 2]
    heights = stats[:, 3]

    expected_area = widths * heights
    area[:, 0] = np.maximum(area[:, 0], 1)
    exp_ratio = expected_area / area[:, 0]
    reject = (exp_ratio > 1 + AREA_THRESHOLD) | (exp_ratio < 1 - AREA_THRESHOLD)
    stats_id = np.concatenate((
        area,
        np.arange(0, retval)[:, None],
    ), axis=1)

    stats_ratio = np.array(sorted(
        stats_id.tolist()
    ), dtype=np.float32)
    if len(stats) <= 2:
        return
    stats_ratio[1:, 0] /= stats_ratio[:-1, 0]
    stats_ratio[0, 0] = 0

    start = None
    end = None

    best = (0, None, None)

    notbad = []

    for i in range(1, retval):
        if start is None and stats_ratio[i, 0] < 1.1:
            start = i - 1
        if end is None and start is not None and stats_ratio[i, 0] > 1.1:
            end = i
            best = max(best, (end - start, start, end))
            if end - start >= LINE_MIN_LENGTH:
                notbad.append((end - start, start, end))
            start = None
            end = None

    new_label = np.zeros_like(labels, np.float32)
    select = []

    length, start, end = best
    if start is None or end is None:
        return

    for length, start, end in notbad:
        for i in range(start, end):
            c = int(stats_ratio[i, 1])
            if reject[c]:
                continue
            std = np.std(warp[labels == c], axis=0)
            if (std > STD_THRESHOLD).any():
                continue
            if (labels == c).sum() < tw * th * MIN_PATCH_RATIO_THRESHOLD:
                continue

            select.append(c)

            new_label[labels == c] = c * 0.02
            cv2.putText(new_label, str(c), tuple([int(i) for i in centroids[c]]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    show_list["canny_labels"] = new_label

    show_list["canny"] = canny
    if len(select) <= 2:
        return

    def get_half(ar):
        xs = np.array(sorted(ar))
        xs = np.array(sorted(xs[1:] - xs[:-1]))
        half = xs[xs > xs.mean()].mean() / 2
        jl = np.zeros(ar.shape, np.int)
        for i in range(ar.shape[0]):
            jl[(ar < ar[i] + half) & (ar > ar[i] - half)] = i

        ms = []
        for i, j in enumerate(sorted(np.unique(jl))):
            ms.append(ar[jl == j].mean())
        ms = sorted(ms)

        return ms, half

    xs, x_half = get_half(centroids[select][:, 0])
    ys, y_half = get_half(centroids[select][:, 1])
    if x_half < tw * MIN_INTERVAL_THRESHOLD or y_half < th * MIN_INTERVAL_THRESHOLD:
        return

    for x in xs:
        cv2.line(new_label, (int(x), 0), (int(x), 999), 0.6)
    for y in ys:
        cv2.line(new_label, (0, int(y)), (999, int(y)), 0.6)

    width = np.median(widths[select])
    height = np.median(heights[select])
    scale = 0.4

    matrix = np.zeros((len(ys), len(xs), 3))
    matrix_use = np.zeros((len(ys), len(xs)))

    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            select_area = np.zeros_like(new_label)
            cv2.circle(select_area, (int(x), int(y)), int(min(width, height) * scale), 255, -1)
            data = warp[select_area > 0]
            std = np.std(data, axis=0)
            mean = np.mean(data, axis=0)
            if (std > STD_THRESHOLD).any():
                continue
            if (mean > 1 - EXPOSURE_THRESHOLD).any():
                continue
            matrix[j, i] = mean
            matrix_use[j, i] = 1

            cv2.circle(new_label, (int(x), int(y)), int(min(width, height) * scale), 0.6, 1)

    show_list["canny_labels"] = new_label
    matrix_show = cv2.resize(matrix, warp.shape[:2], interpolation=cv2.INTER_NEAREST)
    show_list["matrix"] = matrix_show

    best = (-1, None, None, None)

    for i, x in enumerate(xs):
        line = matrix[:, i][matrix_use[:, i] > 0]
        if line.shape[0] < LINE_MIN_LENGTH:
            continue
        diff = line[1:] - line[:-1]
        diff *= 1 if diff.mean() > 0 else - 1
        if diff.min() < -EXPOSURE_THRESHOLD:
            # print(i, x, diff.min(), "<0", line)
            continue
        crstd = np.std(line / line.mean(axis=1)[..., None], axis=0).max()
        if crstd > CRSTD_THRESHOLD:
            continue
        best = max(best, (-crstd, (slice(None, None, None), i), line, diff))
        # print(i, x, crstd, line)

    for j, y in enumerate(ys):
        line = matrix[j][matrix_use[j] > 0]
        if line.shape[0] < LINE_MIN_LENGTH:
            continue
        diff = line[1:] - line[:-1]
        diff *= 1 if diff.mean() > 0 else - 1
        if diff.min() < -EXPOSURE_THRESHOLD:
            # print(j, y, diff.min(), "<0", line)
            continue
        crstd = np.std(line / line.mean(axis=1)[..., None], axis=0).max()
        if crstd > CRSTD_THRESHOLD:
            continue
        best = max(best, (-crstd, (j, slice(None, None, None)), line, diff))
        # print(j, y, crstd, line)

    show_key = np.zeros((len(ys), len(xs)))

    _, idx, line, diff = best
    if idx is None:
        return
    show_key[idx] = 1
    # print("crstd", np.std(line / line.mean(axis=1)[..., None], axis=0).max(), np.std(diff, axis=0))
    # plt.plot(line / line.mean(axis=1)[..., None] / 2)

    good = np.ones(line.shape[0], np.int)
    good[0] = 0
    good[-1] = 0

    for i in range(len(diff)):
        if (diff[i] <= 0).any():
            good[i] = 0
            good[i + 1] = 0

    show_key[idx][matrix_use[idx] > 0] = good * 2

    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            if matrix_use[j, i] > 0:
                if show_key[j, i] == 1:
                    cv2.circle(new_label, (int(x), int(y)), int(min(width, height) * scale), 0.8, 2)
                if show_key[j, i] == 2:
                    cv2.circle(new_label, (int(x), int(y)), int(min(width, height) * scale), 1, 4)

    show_list["canny_labels"] = new_label

    to_be_calibrate = line[good > 0]
    if to_be_calibrate.shape[0] == 0:
        return
    to_be_calibrate = to_be_calibrate.mean(axis=0)
    to_be_list.append(to_be_calibrate)
    if len(to_be_list) > COEFF_LENGTH:
        to_be_list.pop(0)

    new_coeff = 1 / np.mean(to_be_list, axis=0)
    new_coeff /= new_coeff.min()
    warp_calibrated = new_coeff * warp
    show_list["warp_calibrated"] = warp_calibrated

    cv2.waitKey(1)

    bright = warp.mean()

    x = ps[:, 0].mean()
    y = ps[:, 1].mean()

    dist = ((x - raw.shape[1] / 2)**2 + (y - raw.shape[0] / 2)**2)**0.5 / max(raw.shape)

    if ENABLE_PLT:
        plt.scatter(dist, bright)
        plt.draw()
        fit_img = np.fromstring(fit_fig.canvas.tostring_rgb(), dtype=np.uint8, sep='').reshape(fit_fig.canvas.get_width_height()[::-1] + (3,))
        show_list["fit_img"] = fit_img.astype(np.float32) / 255
        # show_list["fit_img"] = cv2.resize(fit_img.astype(np.float32) / 255, (640, 480))

    return new_coeff, [dist.item(), bright.item()]
