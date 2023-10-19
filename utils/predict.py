import pandas as pd
import torch.nn.functional as F
import torch
import os
import numpy as np

def prediction(index, test_loader, model, attributes, opts):
    results = {}
    attribute = attributes
    for batch_idx, (img, image_id_batch) in enumerate(test_loader):
        img = img.cuda()
        with torch.no_grad():
            model.eval()
            outputs = model(img)
            outputs = F.sigmoid(outputs)

            predictions = outputs.cpu().numpy().tolist()

            for i, image_id in enumerate(image_id_batch):
                results[image_id] = predictions[i]


    image_ids = np.array(list(results.keys()))
    preds = np.array(list(results.values()))
    result = np.concatenate([np.expand_dims(image_ids, axis=1), preds], axis=1)
    result_df = pd.DataFrame(result)
    save_path = os.path.join(opts.result_path, f'predictions{index}.csv')
    result_df.to_csv(save_path, header=False, index=False)
    print(f'Prediction completed and results saved to "predictions{index}.csv"')


