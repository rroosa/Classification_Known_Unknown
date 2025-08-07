# Classification_Known_Unknown

This software system code was originally developed as part of the following scientific study:<br> 
<p><b>Rosa Zuccarà, Georgia Fargetta, Alessandro Ortis, Sebastiano Battiato,</b> “Exploiting Adversarial Learning and Topology Augmentation for Open-Set Visual Recognition.” <b>Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops,</b> 2025, pp. 3425-3433.</p>
<hr>
<p>This work aims to modify and enrich the feature space in which classes are represented, thereby improving the separability of known
classes and enhance the model's robustness to out-of-distribution (OOD) inputs. The characteristic descriptor of a sample is defined as
the probability distribution vector produced by the model. The topological enrichment of the feature space is achieved by introducing a
new class, referred to as <i>Neutral</i>, whose ideal descriptor is represented by a uniform distribution, such that it cannot be confidently
associated with any known class, thus inducing maximum uncertainty in the classifier.
The <i>Neutral</i> class is synthetically generated using a custom-designed system that integrates the NEAT technique (NeuroEvolution of
Augmenting Topologies), an evolutionary algorithm for the automatic generation of artificial neural networks. A fitness function, named
F<sub><i>NEAT</i></sub>, has been implemented to guide the evolution of the networks in producing patterns suitable for the intended objective.
The t-SNE technique is used to visualize, in a three-dimensional space, the probability distribution vectors obtained in supervised
classification scenarios, both in the context of closed-set and open-set recognition.</p>

<hr>
<p> In the <b>REPORT</b> folder, there is a document that provides a description and instructions for the execution of the software system. </p>
