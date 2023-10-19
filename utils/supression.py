import os

project_index = os.getcwd().find('fine-grained2019AAAI')
root = os.getcwd()[0:project_index] + 'fine-grained2019AAAI'
import sys

sys.path.append(root)
import torch
from torch import nn
import random


############    可选择的    #####################
class SelectDropMAX(nn.Module):

    def __init__(self, pk=0.5, supression="Join", mask_height=7, mask_width=2):
        """
        - nhận vào một feature map 3 chiều.
        - trả về một danh sách numpy làm mask.
        :param pk: xác suất áp dụng mask ngẫu nhiên cho mỗi feature map
        :param r: chia feature map thành r khối theo chiều dọc
        :param c: chia feature map thành c khối theo chiều ngang
        """
        super(SelectDropMAX, self).__init__()
        self.supression = supression
        self.pk = pk
        self.mask_width = mask_width
        self.mask_height = mask_height

    def helperb1(self, feature_map):
        '''
        - Peak regions suppression
        Phương thức này thực hiện phát hiện các vị trí cực đại trong feature map
        và tạo một mask với giá trị 1 tại các vị trí đó và 0 tại các vị trí còn lại.
        '''
        row, col = torch.where(feature_map == torch.max(feature_map))  # Tìm vị trí giá trị lớn nhất trong feature map
        # Tạo mask có giá trị 1 tại vị trí đó, còn lại là 0
        b1 = torch.zeros_like(feature_map).cuda()
        for i in range(len(row)):
            r, c = int(row[i]), int(col[i])
            b1[r, c] = 1
        return b1  ##tensor

    def create_mask(self, feature_map, mask_height, mask_width, x=None, y=None):
        '''
        Màu đen là 0 và phần còn thiếu của màu trắng là 255.
        Trong sử dụng thực tế, 255 cần phải đổi thành 1. Vì cần phải thêm bớt vào ảnh gốc
        '''
        height, width = feature_map.size()
        mask = torch.zeros_like((feature_map)).cuda()  # Khởi tạo mask toàn 0, kích thước giống feature map
        # Random vị trí và kích thước vùng mask
        mask_x = x if x is not None else random.randint(0, width - mask_width)  # Tọa độ x góc dưới bên trái của phần bị thiếu
        mask_y = y if y is not None else random.randint(0, height - mask_height)  # Tọa độ y của góc trên bên trái của phần bị thiếu
        # Tạo mask bằng cách gán 1 vào vùng ngẫu nhiên
        mask[mask_y:mask_y + mask_height, mask_x:mask_x + mask_width] = 1  #
        return mask  # tensor

    def forward(self, x):
        """
        x---(bs,c,h,w) ([bs, 3, 3, 4])
        """
        batch_supression = []  # Danh sách chứa kết quả sau khi suppress của các sample
        sample_maps_list = torch.split(x, 1)  # Tách dữ liệu đầu vào thành từng mẫu.
        ### 按照batch循环
        for sample_map in sample_maps_list:
            sample_map = sample_map.squeeze(0)  # [1, 3, 3, 4] --> [3, 3, 4]

            ###########   Xử lý từng mẫu  ###########
            if len(sample_map.shape) == 3:
                resb1 = []  # peak mask
                resb2 = []  # random mask
                # tách từng feature map của 1 sample
                feature_maps_list = torch.split(sample_map, 1)  # [1, 3, 4]

                ###   Chuyển qua các kênh
                for feature_map in feature_maps_list:
                    feature_map = feature_map.squeeze(0)  # [1, 3, 4] --> [3, 4]
                    if self.supression == "Peak":
                        tmp = self.helperb1(feature_map)  # peak mask
                        resb1.append(tmp)  # [tensor,tensor..]
                    if self.supression == "Random":
                        tmp1 = self.create_mask(feature_map, self.mask_height, self.mask_width)  # patch mask
                        resb2.append(tmp1)
                    if self.supression == "Join":
                        tmp = self.helperb1(feature_map)  # peak mask
                        resb1.append(tmp)
                        tmp1 = self.create_mask(feature_map, self.mask_height, self.mask_width)  # patch mask
                        resb2.append(tmp1)

                # resb1=torch.stack(resb1,0)
                resb2 = torch.stack(resb2, 0)

            elif len(sample_map.shape) == 2:  # mỗi sample chỉ có 1 feature map
                resb1 = []  # peak mask
                resb2 = []  # random mask
                if self.supression == "Peak":
                    tmp = self.helperb1(sample_map)  # peak mask
                    resb1.append(tmp)

                if self.supression == "Random":
                    tmp1 = self.create_mask(sample_map, self.mask_width, self.mask_height)  # patch mask
                    resb2.append(tmp1)

                if self.supression == "Join":
                    tmp = self.helperb1(sample_map)  # peak mask
                    resb1.append(tmp)

                    tmp1 = self.create_mask(sample_map, self.mask_width, self.mask_height)  # patch mask
                    resb2.append(tmp1)

                # Cần phải sửa đổi thủ công: Peak sử dụng resb1, Random sử dụng resb2 và Join sử dụng cả hai.
                resb1 = torch.stack(resb1, 0)
                resb2 = torch.stack(resb2, 0)

            else:
                raise ValueError

            # Nhân mask với feature map ban đầu để supppress
            # F(s) = F - alpha * (M dot F)
            res_features = []
            if len(sample_map.shape) == 3:
                for x in range(len(resb1)):
                    if self.supression == "Peak":  # lấy mask từ resb1
                        index_block = torch.clamp(resb1[x], 0, 1)  # 每个通道的, giới hạn giá trị của mask từ 0-1
                        res_feature = sample_map[x] - 0.9 * torch.mul(sample_map[x], index_block.cuda())
                        res_features.append(res_feature)
                    if self.supression == "Random":  # lấy mask từ resb2
                        index_block = torch.clamp(resb2[x], 0, 1)  # 每个通道的
                        res_feature = sample_map[x] - 0.9 * torch.mul(sample_map[x], index_block.cuda())
                        res_features.append(res_feature)
                    if self.supression == "Join":  # lấy mask là tổng của resb1 và resb2
                        index_block = torch.clamp(resb1[x] + resb2[x], 0, 1)  # 每个通道的
                        res_feature = sample_map[x] - 0.9 * torch.mul(sample_map[x], index_block.cuda())
                        res_features.append(res_feature)
                res_features = torch.stack(res_features, 0)  # stack tất cả lại thành 1 tensor


            elif len(sample_map.shape) == 2:
                if self.supression == "Peak":
                    index_block = torch.clamp(resb1[x], 0, 1)  # 每个通道的
                    res_features = sample_map[x] - 0.9 * torch.mul(sample_map[x], index_block.cuda())
                if self.supression == "Random":
                    index_block = torch.clamp(resb2[x], 0, 1)  # 每个通道的
                    res_features = sample_map[x] - 0.9 * torch.mul(sample_map[x], index_block.cuda())
                if self.supression == "Join":
                    index_block = torch.clamp(resb1[x] + resb2[x], 0, 1)  # 每个通道的
                    res_features = sample_map[x] - 0.9 * torch.mul(sample_map[x], index_block.cuda())
            batch_supression.append(res_features)
        batch_supression = torch.stack(batch_supression, 0)
        return batch_supression


if __name__ == '__main__':
    feature_maps = torch.rand([16, 3, 3, 4])
    # print("feature maps is: ", feature_maps.size())
    db = SelectDropMAX(mask_height=1, mask_width=2)
    db.cuda()
    res = db(feature_maps.cuda())
    print("################")
    # print(feature_maps)
    # print(res, len(res))





