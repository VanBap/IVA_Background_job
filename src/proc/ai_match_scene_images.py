from dataclasses import dataclass
import cv2
from typing import Tuple, Optional
import numpy as np

from datetime import datetime


RANSAC_THRESHOLD = 0.001
SCALE_TOLERANCE: float = 0.01
ROTATE_TOLERANCE: float = 0.01

GREEN: tuple = (0, 255, 0)
RED: tuple = (0, 0, 255)
BLUE: tuple = (255, 0, 0)
ORANGE: tuple = (0, 165, 255)
WHITE: tuple = (255, 255, 255)

def merge_images(img_1: np.ndarray, img_2: np.ndarray) -> np.ndarray:
    """Merge two images side by side with annotations.
    Args:
        img_1: First image
        img_2: Second image
    Returns:
        Combined visualization image
    """
    cv2.putText(img_1, "Background", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    h_1, w_1 = img_1.shape[:2]
    h_2, w_2 = img_2.shape[:2]

    viz = np.zeros((max(h_1, h_2), w_1 + w_2, 3), dtype="uint8")
    viz[0:h_1, 0:w_1] = img_1
    viz[0:h_2, w_1:] = img_2

    return viz

def visualize_matching(img_1: np.ndarray,
                       img_2: np.ndarray,
                       kps_1: np.ndarray,
                       kps_2: np.ndarray,
                       matches: list,
                       h_matrix: np.ndarray,
                       status: np.ndarray) -> np.ndarray:
    """
    Visualize matching points and transformations between images.
    Args:
        img_1: First image
        img_2: Second image
        kps_1: Keypoints from first image
        kps_2: Keypoints from second image
        matches: List of matching point pairs
        h_matrix: Homography matrix
        status: Inlier/outlier status for matches
    Returns:
        Visualization image with matches and transformations
    """
    viz = merge_images(img_1, img_2)
    h_1, w_1 = img_1.shape[:2]
    h_2, w_2 = img_2.shape[:2]

    corners_1 = np.array([(0, 0), (1, 0), (1, 1), (0, 1)], dtype="float32")
    corners_2 = (cv2.perspectiveTransform(np.array([corners_1]), h_matrix)[0] * np.array([w_2, h_2])).astype(np.int32)

    for pt in corners_1:
        cv2.circle(viz[0:h_1, 0:w_1], (int(pt[0] * w_1), int(pt[1] * h_1)), 5, RED, 5, cv2.LINE_AA)

    for pt in corners_2:
        cv2.circle(viz[0:h_2, w_1:], pt, 5, RED, 5, cv2.LINE_AA)

    for i in range(-1, 3):
        cv2.line(viz[0:h_2, w_1:], corners_2[i], corners_2[i + 1], GREEN, 2, cv2.LINE_AA)

    for (train_idx, query_idx), s in zip(matches, status):
        if s == 1:
            pt1 = (int(kps_1[query_idx][0]), int(kps_1[query_idx][1]))
            pt2 = (int(kps_2[train_idx][0]) + img_1.shape[1], int(kps_2[train_idx][1]))

            cv2.line(viz, pt1, pt2, BLUE, 1, cv2.LINE_AA)
            cv2.circle(viz, pt1, 5, RED, 1, cv2.LINE_AA)
            cv2.circle(viz, pt2, 5, RED, 1, cv2.LINE_AA)

    return viz


@dataclass
class SceneChangeResult:
    """
    Represents the result of a scene matching operation.
    """
    is_matched: bool = True
    time_stamp: Optional[datetime] = None
    num_matches: Optional[int] = 0
    scale: Optional[float] = 1.0
    rotation: Optional[float] = 0.0
    h_matrix: Optional[np.ndarray] = None
    camera_image: Optional[np.ndarray] = None
    viz: Optional[np.ndarray] = None

class SceneMatcher:
    """
    Class to detect scene changes between two images.
    """

    def __init__(self, visualize: bool = False):
        self.sift = cv2.SIFT_create()
        self.visualize = visualize
        self.matcher = cv2.DescriptorMatcher.create("BruteForce")

    def _extract_features(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract SIFT features from image.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kps, descriptors = self.sift.detectAndCompute(gray, None)
        return np.float32([kp.pt for kp in kps]), descriptors

    def _match_features(self, desc_1: np.ndarray, desc_2: np.ndarray) -> list:
        """
        Match features between two images.
        """
        raw_matches = self.matcher.knnMatch(desc_1, desc_2, 2)
        return [(m[0].trainIdx, m[0].queryIdx) for m in raw_matches
                if len(m) == 2 and m[0].distance < m[1].distance * 0.75]

    def match_scenes(self, img_1: np.ndarray, img_2: np.ndarray) -> bool:
        """
        Perform scene matching between two images.
        """
        _, descriptors_1 = self._extract_features(img_1)
        _, descriptors_2 = self._extract_features(img_2)
        matches = self._match_features(descriptors_1, descriptors_2)

        if len(matches) <= 100:
            return False
        return True

    def _compute_homography(self,
                            img_1: np.ndarray,
                            img_2: np.ndarray,
                            kps_1: np.ndarray,
                            kps_2: np.ndarray,
                            matches: list) -> SceneChangeResult:
        """
        Compute homography matrix and create visualization.
        """
        shape_wh_1 = img_1.shape[:2][::-1]
        shape_wh_2 = img_2.shape[:2][::-1]

        pts_1 = np.float32([kps_1[i] for _, i in matches]) / shape_wh_1
        pts_2 = np.float32([kps_2[i] for i, _ in matches]) / shape_wh_2
        h_matrix, status = cv2.findHomography(pts_1,
                                              pts_2,
                                              cv2.RANSAC,
                                              RANSAC_THRESHOLD)

        if h_matrix is None:
            return SceneChangeResult(
                is_matched=False,
                num_matches=len(matches),
                viz=merge_images(img_1, img_2) if self.visualize else None,
                time_stamp=datetime.now(),
                camera_image=img_2
            )

        result = SceneChangeResult(is_matched=True,
                                   num_matches=len(matches),
                                   h_matrix=h_matrix,
                                   scale=np.sqrt(h_matrix[0, 0] ** 2 + h_matrix[1, 0] ** 2),
                                   rotation=np.arctan2(h_matrix[1, 0], h_matrix[0, 0]),
                                   camera_image=img_2,
                                   time_stamp=datetime.now())

        if abs(result.scale - 1) > SCALE_TOLERANCE or abs(result.rotation) > ROTATE_TOLERANCE:
            result.is_matched = False

        if self.visualize:
            result.viz = visualize_matching(img_1=img_1,
                                            img_2=img_2,
                                            kps_1=kps_1,
                                            kps_2=kps_2,
                                            matches=matches,
                                            h_matrix=h_matrix,
                                            status=status)
        return result



