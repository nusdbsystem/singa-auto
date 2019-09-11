from lime import lime_image
import torchvision.transforms as transforms
from skimage.segmentation import mark_boundaries
from skimage import io
class Lime():
    """
    Explain machine learning classifiers using lime
    """
    def __init__(self, model):
        self.model = model
        self.num_samples = 500
        self.sett = 1000
    
    def get_pil_transform(): 
        transf = transforms.Compose([transforms.Resize((512, 512))])    
        return transf

    def get_preprocess_transform():
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])     
        transf = transforms.Compose([transforms.ToTensor(),normalize])   
        return transf

    def batch_predict(images):
        self.model.eval()
        batch = torch.stack(tuple(preprocess_transform(i) for i in images), dim=0)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        batch = batch.to(device)
        logits = self.model(batch)
        out = torch.sigmoid(logits).data.cpu().numpy()
        #tmp = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
        #tmp.remove(obser_index)
        #out = np.delete(out, tmp, axis=1)
        return out

    def Explain(self, img):
        test_pred = batch_predict([pill_transf(img)])
        explainer = lime_image.LimeImageExplainer()
        explanation = explainer.explain_instance(np.array(pill_transf(img)), batch_predict, top_labels=5, hide_color=0, num_samples=1000)
        temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=False)

        img_boundry1 = mark_boundaries(temp/255.0, mask)
        

        io.imsave("./lime_image/" + image_name.split(".")[0]+"_lime_top5.png",img_boundry1)
        io.imsave("./lime_image/" + image_name.split(".")[0]+"_lime_top1_resnet.png",img_boundry1)

