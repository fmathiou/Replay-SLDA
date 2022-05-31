# Combining Rehearsal with Linear Discriminant Analysis in Online Continual Learning
This is a PyTorch implementation of the Replay-SLDA algorithm developed for my master's thesis. It is an extension of previous work by [Hayes and Kanan (2020)](https://openaccess.thecvf.com/content_CVPRW_2020/html/w15/Hayes_Lifelong_Machine_Learning_With_Deep_Streaming_Linear_Discriminant_Analysis_CVPRW_2020_paper.html) on Deep Streaming Linear Discriminant Analysis (SLDA). Replay-SLDA alleviates the requirement for a pre-trained and frozen feature extractor, allowing the network to learn new feature representations. The algorithm uses rehearsal to continually train the feature extractor while updating the classification layer through linear discriminant analysis. The code here contains the final implementation of the model, referred to as Replay-SLDA (v2) in the main text. This model was tested on three datasets ([CIFAR-10, CIFAR-100, ](https://www.cs.toronto.edu/~kriz/cifar.html)[Histology (CRH)](https://www.nature.com/articles/srep27988)) and achieved higher final accuracy compared to SLDA already from memory sizes corresponding on average to 20 samples per class. The complete thesis text can be found [here](https://repository-teneo-libis-be.kuleuven.e-bronnen.be/delivery/DeliveryManagerServlet?dps_pid=IE16474940&).
## Environments
- Python=3.9.2
- PyTorch=1.8.1
- Torchvision=0.9.1
- NumPy=1.22.4
## Usage
Set up relevent parameters and run main.py. Different training algorithms can be imported from training.py.
## Results on CIFAR-10 
<img src="./Images/cifar10_comparison.jpg" width=50% height=50%>

## Results on CIFAR-100 
<img src="./Images/cifar100_comparison.jpg" width=50% height=50%>

## Results on CRH 
<img src="./Images/histology_comparison.jpg" width=50% height=50%>
