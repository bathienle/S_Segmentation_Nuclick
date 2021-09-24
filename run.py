# -*- coding: utf-8 -*-

# * Copyright (c) 2009-2021. Authors: see NOTICE file.
# *
# * Licensed under the Apache License, Version 2.0 (the "License");
# * you may not use this file except in compliance with the License.
# * You may obtain a copy of the License at
# *
# *      http://www.apache.org/licenses/LICENSE-2.0
# *
# * Unless required by applicable law or agreed to in writing, software
# * distributed under the License is distributed on an "AS IS" BASIS,
# * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# * See the License for the specific language governing permissions and
# * limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import cv2
from shapely.wkt import load
import torch
import os

from pathlib import Path
from PIL import Image

from shapely.affinity import affine_transform
from shapely.geometry import Polygon

from torchvision.transforms.functional import to_tensor

from cytomine import CytomineJob
from cytomine.models import Annotation, AnnotationCollection

from src import NuClick, post_process


__author__ = "LE Ba Thien <ba.le@uliege.be>"


def load_model(filename):
    model = NuClick()

    weight_path = os.path.join('/weights/', filename)

    model.load_state_dict(torch.load(weight_path, map_location='cpu'))
    model.eval()

    return model


def predict(model, image):
    shape = (image.shape[0], 2, image.shape[2], image.shape[3])
    signal = torch.zeros(shape, device=image.device)
    inputs = torch.cat([image, signal], dim=1)

    with torch.no_grad():
        preds = model(inputs)

    return preds


def main(argv):
    with CytomineJob.from_cli(argv) as cj:
        cj.job.update(progress=0, statusComment="Initialisation")

        # Create the image directory
        working_path = str(Path.home())
        images_path = os.path.join(working_path, 'images')
        os.makedirs(images_path, exist_ok=True)

        # Load the NuClick model
        model = load_model(f'weight-{cj.parameters.type}.pth')

        cj.job.update(progress=15, statusComment="Load the model")

        # Get the images of the project
        images = [
            int(id) for id in cj.parameters.cytomine_id_images.split(',')
        ]

        # Get the ROIs
        paths = []
        for image in images:
            roi_annotations = AnnotationCollection(
                project=cj.parameters.cytomine_id_project,
                term=cj.parameters.roi_id_term,
                image=image,
                showWKT=True
            )
            roi_annotations.fetch()

            # Download the ROIs of the targeted project
            for roi in roi_annotations:
                filename = f'{roi.image}-{roi.id}.png'
                filepath = os.path.join(images_path, filename)
                roi.dump(dest_pattern=filepath)
                paths.append((roi.image, filepath))

        cj.job.update(progress=40, statusComment="Fetch the ROIs")

        # Predict the object inside the ROI
        annotations = []
        for image_id, path in paths:
            # Open the ROI image
            image = to_tensor(Image.open(path))

            # Predictions
            preds = predict(model, image.unsqueeze(0))

            masks = post_process(
                preds,
                min_size=cj.parameters.min_size,
                area_threshold=cj.parameters.area_threshold
            )

            contours, _ = cv2.findContours(
                masks,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            contours = [c for c in contours if cv2.contourArea(c) > 100]

            for c in contours:
                mask = Polygon(c)
                annotation = Annotation(
                    location=mask.wkt,
                    id_image=image_id,
                    id_project=cj.parameters.cytomine_id_project
                )
                annotations.append(annotation)

        cj.job.update(progress=90, statusComment="Predictions done")

        # Save all the annotations
        annotations.save()

        cj.job.update(progress=100, statusComment="Job terminated")


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
