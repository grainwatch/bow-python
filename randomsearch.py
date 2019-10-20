import models
from dataset import objectdataset

def main():
    global train_imgs, train_rects, train_labels
    global test_imgs, test_rects, test_labels
    (train_imgs, train_rects, train_labels), (test_imgs, test_rects, test_labels) = objectdataset.load_grain_dataset('C:\\Users\\Alex\\Desktop\\fulldataset', ['corn'])
    #search_hog_params()
    search_db_scan_params()

def search_hog_params():
    blocks_grid = [(1, 1), (2, 2), (4, 4), (8, 8), (16, 16)]
    cells_per_block_grid = [(1, 1), (2, 2), (4, 4), (8, 8), (16, 16)]
    block_stride_in_cells_grid = [(1, 1), (2, 2), (3, 3)]
    best_ap, best_blocks, same_blocks = search_param(hog_eval_blocks_param, blocks_grid)
    best_ap, best_cells, same_cells = search_param(hog_eval_cells_param, cells_per_block_grid)
    best_ap, best_block_stride, same_block_strides = search_param(hog_eval_block_stride_param, block_stride_in_cells_grid)
    print('lol')

def search_db_scan_params():
    standard_lower_bound = (0, 150, 150)
    standard_upper_bound = (50, 255, 255)
    lower_bound_0_grid = [0, 5, 10, 15, 20, 25]
    upper_bound_0_grid = [40, 45, 50, 55, 60, 65]
    lower_bound_grid = list(map(lambda x0: (x0, standard_lower_bound[1], standard_lower_bound[2]), lower_bound_0_grid))
    upper_bound_grid = list(map(lambda x0: (x0, standard_upper_bound[1], standard_upper_bound[2]), upper_bound_0_grid))
    minpts_grid = [5, 25, 45, 65, 85, 105]
    minpts_small_grid = [55, 60, 65, 70, 75, 80]
    eps_grid = [20, 40, 60, 80, 100]
    eps_small_grid = [30, 35, 40, 45, 50]
    best_ap, best_minpts, same_minpts = search_param(color_dbscan_eval_minpts_param, minpts_small_grid)
    print(f'best minpts: {best_minpts} with {best_ap} ap')
    print(same_minpts)
    best_ap, best_eps, same_eps = search_param(color_dbscan_eval_eps_param, eps_small_grid)
    print(f'best eps: {best_eps} with {best_ap} ap')
    best_ap, best_lower, same_eps = search_param(color_dbscan_eval_lower_param, lower_bound_grid)
    print(f'best lower: {best_lower} with {best_ap} ap')
    best_ap, best_upper, same_eps = search_param(color_dbscan_eval_upper_param, upper_bound_grid)
    print(f'best upper: {best_upper} with {best_ap} ap')
    print(same_eps)


def hog_eval_blocks_param(blocks):
    hog = models.HOGSlidingWindowDetector(blocks=blocks)
    return eval_learning_objdetector(hog)

def hog_eval_cells_param(cells):
    hog = models.HOGSlidingWindowDetector(cells_per_block=cells)
    return eval_learning_objdetector(hog)

def hog_eval_block_stride_param(block_stride):
    hog = models.HOGSlidingWindowDetector(block_stride_in_cells=block_stride)
    return eval_learning_objdetector(hog)

def color_dbscan_eval_lower_param(lower):
    color_dbcan = models.ColorDBScan(1, lower_bounds=lower)
    return eval_objdetector(color_dbcan)

def color_dbscan_eval_upper_param(upper):
    color_dbcan = models.ColorDBScan(1, upper_bounds=upper)
    return eval_objdetector(color_dbcan)

def color_dbscan_eval_minpts_param(minpts):
    color_dbcan = models.ColorDBScan(1, minpts=minpts)
    return eval_objdetector(color_dbcan)

def color_dbscan_eval_eps_param(eps):
    color_dbcan = models.ColorDBScan(1, eps=eps)
    return eval_objdetector(color_dbcan)

def eval_learning_objdetector(l_objdetector):
    l_objdetector.fit(train_imgs, train_rects, train_labels)
    ap, confusion_matrix, all_iou_sum = l_objdetector.evaluate(test_imgs, test_rects, test_labels)
    return ap

def eval_objdetector(color_dbcan):
    ap, confusion_matrix, all_iou_sum = color_dbcan.evaluate(test_imgs, test_rects, test_labels)
    return ap

def search_param(eval_param_func, param_grid):
    best_param = None
    best_score = 1
    param_with_same_scores = []
    for param in param_grid:
        score = eval_param_func(param)
        if score < best_score:
            best_param = param
            best_score = score
        elif score == best_score:
            param_with_same_scores.append((best_param, param))
            best_param = param
        print('evaluated param')
    return best_score, best_param, param_with_same_scores

if __name__ == '__main__':
    main()

