import numpy as np




def sort_labels(list_labels):
    """
    function to sort labels
    @return list of sorted labels
    """
    sorted_labels = []
    for list_label in list_labels :
        sorted_label = sorted(list_label[1])
        sorted_labels.append(sorted_label)
    
    return sorted_labels

def additive_vote(result_fcm, result_gaussian, result_dbscan, label_fcm, label_gauss, label_db, number_of_cluster, removed_classes=[]):
    """
    function to calculate additive vote
    """
    fcm_sorted, gauss_sorted, db_sorted = sort_labels([label_fcm, label_gauss, label_db])
    final_result = np.zeros((result_fcm.shape[0],result_fcm.shape[1],3), dtype=np.uint8)
    for i in range(result_fcm.shape[0]):
        for j in range(result_fcm.shape[1]):
            break_loop = False
            for k in range(number_of_cluster):
                # uncomment to remove background from decision
                for to_remove in removed_classes :
                    if (result_fcm[i][j] == label_fcm[0][label_fcm[1].index(fcm_sorted[to_remove])] ):
                        break_loop = True
                if break_loop :
                    break
                if (result_gaussian[i][j] != label_gauss[0][label_gauss[1].index(gauss_sorted[k])] and
                            result_dbscan[i][j] != label_db[0][label_db[1].index(db_sorted[k])] 
                            and (result_fcm[i][j] == label_fcm[0][label_fcm[1].index(fcm_sorted[k])])) :
                    final_result[i][j] = np.array((255,0,0))
                    break
                if (result_gaussian[i][j] != label_gauss[0][label_gauss[1].index(gauss_sorted[k])] and
                            result_dbscan[i][j] == label_db[0][label_db[1].index(db_sorted[k])] 
                            and (result_fcm[i][j] != label_fcm[0][label_fcm[1].index(fcm_sorted[k])])) :
                    final_result[i][j] = np.array((255,0,0))
                    break
                if (result_gaussian[i][j] == label_gauss[0][label_gauss[1].index(gauss_sorted[k])] and
                            result_dbscan[i][j] != label_db[0][label_db[1].index(db_sorted[k])] 
                            and (result_fcm[i][j] != label_fcm[0][label_fcm[1].index(fcm_sorted[k])])) :
                    final_result[i][j] = np.array((255,0,0))
                    break
                if ((result_gaussian[i][j] == label_gauss[0][label_gauss[1].index(gauss_sorted[k])]) and
                            (result_dbscan[i][j] == label_db[0][label_db[1].index(db_sorted[k])]) 
                            and (result_fcm[i][j] == label_fcm[0][label_fcm[1].index(fcm_sorted[k])])):
                    final_result[i][j] = np.array((0,255,0))
                    break

                elif ((result_gaussian[i][j] == label_gauss[0][label_gauss[1].index(gauss_sorted[k])] and
                            result_dbscan[i][j] == label_db[0][label_db[1].index(db_sorted[k])] 
                            and result_fcm[i][j] != label_fcm[0][label_fcm[1].index(fcm_sorted[k])])):
                    final_result[i][j] = np.array((236,243,30))
                    break

                elif (result_gaussian[i][j] == label_gauss[0][label_gauss[1].index(gauss_sorted[k])] and
                            result_dbscan[i][j] != label_db[0][label_db[1].index(db_sorted[k])]
                            and result_fcm[i][j] == label_fcm[0][label_fcm[1].index(fcm_sorted[k])]) :
                    final_result[i][j] = np.array((236,243,30))
                    break

                elif (result_gaussian[i][j] != label_gauss[0][label_gauss[1].index(gauss_sorted[k])] and
                            result_dbscan[i][j] == label_db[0][label_db[1].index(db_sorted[k])] 
                            and (result_fcm[i][j] == label_fcm[0][label_fcm[1].index(fcm_sorted[k])])) :
                    final_result[i][j] = np.array((236,243,30))
                    break 
    return final_result

def crop_class(label_matrix, labels, targted_class, color=(255,0,0)):
    label_sorted = sort_labels([labels])[0]
    class_result = np.zeros((label_matrix.shape[0],label_matrix.shape[1],3), dtype=np.uint8)
    for i in range(label_matrix.shape[0]):
        for j in range(label_matrix.shape[1]):
            if label_matrix[i][j] == labels[0][labels[1].index(label_sorted[targted_class])]:
                class_result[i][j] = np.array(color)
    return class_result

def additive_all(result_fcm, result_gaussian, result_dbscan, label_fcm, label_gauss, label_db, number_of_cluster):
    """
    function to calculate additive vote
    """
    fcm_sorted, gauss_sorted, db_sorted = sort_labels([label_fcm, label_gauss, label_db])
    final_result = np.zeros((result_fcm.shape[0],result_fcm.shape[1],3), dtype=np.uint8)
    for i in range(result_fcm.shape[0]):
        for j in range(result_fcm.shape[1]):
            for k in range(number_of_cluster):
                if result_fcm[i][j] == label_fcm[0][label_fcm[1].index(fcm_sorted[2])] :
                    break
                if ((result_gaussian[i][j] == label_gauss[0][label_gauss[1].index(gauss_sorted[k])]) and
                            (result_dbscan[i][j] == label_db[0][label_db[1].index(db_sorted[k])]) 
                            and (result_fcm[i][j] == label_fcm[0][label_fcm[1].index(fcm_sorted[k])])):
                    final_result[i][j] = np.array((0,255,0))
                    break

                elif ((result_gaussian[i][j] == label_gauss[0][label_gauss[1].index(gauss_sorted[k])] and
                            result_dbscan[i][j] == label_db[0][label_db[1].index(db_sorted[k])] 
                            and result_fcm[i][j] != label_fcm[0][label_fcm[1].index(fcm_sorted[k])])):
                    final_result[i][j] = np.array((236,243,30))
                    break

                elif (result_gaussian[i][j] == label_gauss[0][label_gauss[1].index(gauss_sorted[k])] and
                            result_dbscan[i][j] != label_db[0][label_db[1].index(db_sorted[k])]
                            and result_fcm[i][j] == label_fcm[0][label_fcm[1].index(fcm_sorted[k])]) :
                    final_result[i][j] = np.array((236,243,30))
                    break

                elif (result_gaussian[i][j] != label_gauss[0][label_gauss[1].index(gauss_sorted[k])] and
                            result_dbscan[i][j] == label_db[0][label_db[1].index(db_sorted[k])] 
                            and (result_fcm[i][j] == label_fcm[0][label_fcm[1].index(fcm_sorted[k])])) :
                    final_result[i][j] = np.array((236,243,30))
                    break 
    return final_result

def generate_mask(result_fcm, result_gaussian, result_dbscan, label_fcm, label_gauss, label_db, class_index):
    fcm_sorted, gauss_sorted, db_sorted = sort_labels([label_fcm, label_gauss, label_db])
    mask = np.zeros((result_fcm.shape[0],result_fcm.shape[1]), dtype=np.uint8)
    for i in range(result_fcm.shape[0]):
            for j in range(result_fcm.shape[1]):                              
                if ((result_gaussian[i][j] == label_gauss[0][label_gauss[1].index(gauss_sorted[class_index])]) and
                                (result_dbscan[i][j] == label_db[0][label_db[1].index(db_sorted[class_index])]) 
                                and (result_fcm[i][j] == label_fcm[0][label_fcm[1].index(fcm_sorted[class_index])])):
                    mask[i][j] = 1
    return mask


def mask_image(mask, image):
    crop = np.zeros((mask.shape[0],mask.shape[1],3), dtype=np.uint8)
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):   
            if mask[i][j] == 1:
                crop[i][j] = image[i][j]
    return crop