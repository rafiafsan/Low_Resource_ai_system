import math
import cv2
from visualization.dashboard import draw_dashboard_bag_associator


class BagPersonAassociator():

    def __init__(self, min_consecutive_frames=5, max_bag_distance=130, min_iob=0.10, association_threshold=0.45):
        # Configuration
        self.MIN_CONSECUTIVE_FRAMES = min_consecutive_frames
        self.MAX_BAG_DISTANCE = max_bag_distance
        self.MIN_IOB = min_iob
        self.ASSOCIATION_THRESHOLD = association_threshold
        
        self.person_bag_candidates = {}     #person and bag track id in each frames
        self.person_bag_associations = {}   #bag id corresponding to the person id(association)
        self.used_person_ids = set()        
        self.used_bag_ids = set()
        self.person_with_bag_count = 0

    def evict_lost_tracks(self, lost_ids):
        """Evict lost tracks to prevent unbounded memory growth."""
        if not lost_ids:
            return
        for tid in lost_ids:
            self.person_bag_associations.pop(tid, None)
            self.used_person_ids.discard(tid)
            candidate_keys = [k for k in self.person_bag_candidates if k[0] == tid or k[1] == tid]
            for k in candidate_keys:
                self.person_bag_candidates.pop(k, None)

    def associate(self, detections, bag_detections):

        results = []

        for person in detections:

            person_id = person["track_id"]
            px1, py1, px2, py2 = person["bbox"]
            person_midpoint = person["person_lower_body_midpoint"]

            if person_id in self.used_person_ids:
                continue

            best_bag = None
            best_score = 0

            for bag in bag_detections:

                bag_id = bag["track_id"]
                bx1, by1, bx2, by2 = bag["bbox"]
                bag_midpoint = bag["bag_midpoint"]


                if bag_id in self.used_bag_ids:
                    continue

                # Intersection over Bag area
                ix1 = max(px1, bx1)
                iy1 = max(py1, by1)

                ix2 = min(px2, bx2)
                iy2 = min(py2, by2)

                intersection_width = max( 0, ix2 - ix1 )
                intersection_height = max( 0, iy2 - iy1 )

                intersection_area = ( intersection_width * intersection_height )
                bag_area = max( 1, (bx2 - bx1) * (by2 - by1) )
                iob = ( intersection_area / bag_area )
                

                # 2. Distance
                px, py = person_midpoint
                bx, by = bag_midpoint
                distance = math.sqrt( (px - bx) ** 2 + (py - by) ** 2 )
                
                # 3. Bag position relative to person
                person_width = max( 1, px2 - px1 )
                person_height = max(1, py2 - py1 )

                relative_x = (bx - px1) / person_width
                relative_y = (by - py1) / person_height
                position_score = 0.0

                # Bag is horizontally close to person
                if 0 <= relative_x <= 1.5: position_score += 0.5
                # Bag is around lower part of person
                if relative_y >= 0.45: position_score += 0.5

                # 4. Distance score
                if distance <= self.MAX_BAG_DISTANCE:
                    distance_score = (
                        1 - distance / self.MAX_BAG_DISTANCE
                    )
                else:
                    distance_score = 0.0

                # 5. IoB score
                if iob >= self.MIN_IOB:
                    iob_score = min( iob, 1.0 )
                else:
                    iob_score = 0.0

                # 6. FINAL ASSOCIATION SCORE
                association_score = ( 0.4 * iob_score + 0.4 * distance_score + 0.2 * position_score )
                
                # Candidate validation
                valid_candidate = (
                    distance <= self.MAX_BAG_DISTANCE
                    and
                    (
                        iob >= self.MIN_IOB
                        or
                        position_score >= 0.50
                    )
                    and
                    association_score >=
                    self.ASSOCIATION_THRESHOLD
                )

                if not valid_candidate:
                    continue

                # Keep best bag for this person
                if association_score > best_score:
                    best_score = association_score
                    best_bag = bag

            # TEMPORAL ASSOCIATION
            if best_bag is None:
                continue

            bag_id = best_bag["track_id"]

            association_key = ( person_id, bag_id )

            # Increase consecutive frame count
            self.person_bag_candidates[association_key] = (
                self.person_bag_candidates.get(
                    association_key, 0
                    ) + 1
                    #if found the association key then increment the count by 1
                    #if not found then set the count to 0 and increment by 1
            )
            consecutive_frames = (
                self.person_bag_candidates[
                    association_key
                ]
            )

            # CONFIRM AFTER N CONSECUTIVE FRAMES
            if (consecutive_frames >=self.MIN_CONSECUTIVE_FRAMES):
                if ( person_id in self.used_person_ids ):
                    continue
                if ( bag_id in self.used_bag_ids ):
                    continue

                # Save association
                self.person_bag_associations[person_id] = bag_id

                # Mark IDs as used
                self.used_person_ids.add( person_id )
                self.used_bag_ids.add( bag_id )

                # Increment counter
                self.person_with_bag_count += 1

                result = {
                    "person_with_bag": self.person_with_bag_count,
                    # "person_track_id": person_id,
                    # "bag_track_id": bag_id,
                    "score": round( best_score, 2 )
                }

                results.append(result)

                # Remove candidate histories involving
                # this person or bag
                keys_to_remove = []

                for key in self.person_bag_candidates:
                    p_id, b_id = key
                    if ( p_id == person_id or b_id == bag_id ):
                        keys_to_remove.append( key )

                for key in keys_to_remove:
                    del self.person_bag_candidates[ key ]

                print("TOTAL:", self.person_with_bag_count)

        return results

    def associate_person_bag_exit(self, detections, bag_detections):
    
            results = []
    
            for person in detections:
    
                person_id = person["track_id"]
                px1, py1, px2, py2 = person["bbox"]
                person_midpoint = person["bottom_center"]
    
                if person_id in self.used_person_ids:
                    continue
    
                best_bag = None
                best_score = 0
    
                for bag in bag_detections:
    
                    bag_id = bag["track_id"]
                    bx1, by1, bx2, by2 = bag["bbox"]
                    bag_midpoint = bag["bag_midpoint"]
    
    
                    if bag_id in self.used_bag_ids:
                        continue
    
                    # Intersection over Bag area
                    ix1 = max(px1, bx1)
                    iy1 = max(py1, by1)
    
                    ix2 = min(px2, bx2)
                    iy2 = min(py2, by2)
    
                    intersection_width = max( 0, ix2 - ix1 )
                    intersection_height = max( 0, iy2 - iy1 )
    
                    intersection_area = ( intersection_width * intersection_height )
                    bag_area = max( 1, (bx2 - bx1) * (by2 - by1) )
                    iob = ( intersection_area / bag_area )
                    
    
                    # 2. Distance
                    px, py = person_midpoint
                    bx, by = bag_midpoint
                    distance = math.sqrt( (px - bx) ** 2 + (py - by) ** 2 )
                    
                    # 3. Bag position relative to person
                    person_width = max( 1, px2 - px1 )
                    person_height = max(1, py2 - py1 )
    
                    relative_x = (bx - px1) / person_width
                    relative_y = (by - py1) / person_height
                    position_score = 0.0
    
                    # Bag is horizontally close to person
                    if 0 <= relative_x <= 1.5: position_score += 0.5
                    # Bag is around lower part of person
                    if relative_y >= 0.45: position_score += 0.5
    
                    # 4. Distance score
                    if distance <= self.MAX_BAG_DISTANCE:
                        distance_score = (
                            1 - distance / self.MAX_BAG_DISTANCE
                        )
                    else:
                        distance_score = 0.0
    
                    # 5. IoB score
                    if iob >= self.MIN_IOB:
                        iob_score = min( iob, 1.0 )
                    else:
                        iob_score = 0.0
    
                    # 6. FINAL ASSOCIATION SCORE
                    association_score = ( 0.4 * iob_score + 0.4 * distance_score + 0.2 * position_score )
                    
                    # Candidate validation
                    valid_candidate = (
                        distance <= self.MAX_BAG_DISTANCE
                        and
                        (
                            iob >= self.MIN_IOB
                            or
                            position_score >= 0.50
                        )
                        and
                        association_score >=
                        self.ASSOCIATION_THRESHOLD
                    )
    
                    if not valid_candidate:
                        continue
    
                    # Keep best bag for this person
                    if association_score > best_score:
                        best_score = association_score
                        best_bag = bag
    
                # TEMPORAL ASSOCIATION
                if best_bag is None:
                    continue
    
                bag_id = best_bag["track_id"]
    
                association_key = ( person_id, bag_id )
    
                # Increase consecutive frame count
                self.person_bag_candidates[association_key] = (
                    self.person_bag_candidates.get(
                        association_key, 0
                        ) + 1
                        #if found the association key then increment the count by 1
                        #if not found then set the count to 0 and increment by 1
                )
                consecutive_frames = (
                    self.person_bag_candidates[
                        association_key
                    ]
                )
    
                # CONFIRM AFTER N CONSECUTIVE FRAMES
                if (consecutive_frames >=self.MIN_CONSECUTIVE_FRAMES):
                    if ( person_id in self.used_person_ids ):
                        continue
                    if ( bag_id in self.used_bag_ids ):
                        continue
    
                    # Save association
                    self.person_bag_associations[person_id] = bag_id
    
                    # Mark IDs as used
                    self.used_person_ids.add( person_id )
                    self.used_bag_ids.add( bag_id )
    
                    # Increment counter
                    self.person_with_bag_count += 1
    
                    result = {
                        "person_with_bag": self.person_with_bag_count,
                        "person_track_id": person_id,
                        "bag_track_id": bag_id,
                        "score": round( best_score, 2 )
                    }
    
                    results.append(result)
    
                    # Remove candidate histories involving
                    # this person or bag
                    keys_to_remove = []
    
                    for key in self.person_bag_candidates:
                        p_id, b_id = key
                        if ( p_id == person_id or b_id == bag_id ):
                            keys_to_remove.append( key )
    
                    for key in keys_to_remove:
                        del self.person_bag_candidates[ key ]
    
                    print("TOTAL:", self.person_with_bag_count)
    
            return results
    
    
    # GET ALL CONFIRMED ASSOCIATIONS

    def get_associations(self):
        return self.person_bag_associations.copy()

    # GET TOTAL PERSONS WITH BAGS
    def get_count(self):
        return self.person_with_bag_count

    # RESET
    def reset(self):

        self.person_bag_candidates.clear()
        self.person_bag_associations.clear()
        self.used_person_ids.clear()
        self.used_bag_ids.clear()
        self.person_with_bag_count = 0

    def draw_confirmed_associations(self, frame, detections, bag_detections):


    # Go through all confirmed person -> bag associations
     for person_id, bag_id in self.person_bag_associations.items():

        person_data = None
        bag_data = None

        # Find the corresponding person detection
        for person in detections:
            if person["track_id"] == person_id:
                person_data = person
                break

        # Find the corresponding bag detection
        for bag in bag_detections:
            if bag["track_id"] == bag_id:
                bag_data = bag
                break

        # If either object is not currently visible, skip drawing
        if person_data is None or bag_data is None:
            continue

        # Person bounding box
        px1, py1, px2, py2 = map(
            int,
            person_data["bbox"]
        )

        # Bag bounding box
        bx1, by1, bx2, by2 = map(
            int,
            bag_data["bbox"]
        )

        # Draw person box - GREEN
        cv2.rectangle(
            frame,
            (px1, py1),
            (px2, py2),
            (0, 255, 0),
            2
        )

        # Draw bag box - GREEN
        cv2.rectangle(
            frame,
            (bx1, by1),
            (bx2, by2),
            (0, 255, 0),
            2
        )

        # Get midpoints
        person_midpoint = person_data["person_lower_body_midpoint"]
        bag_midpoint = bag_data["bag_midpoint"]

        px, py = map(int, person_midpoint)
        bx, by = map(int, bag_midpoint)

        # Draw midpoint points
        cv2.circle(
            frame,
            (px, py),
            5,
            (0, 255, 0),
            -1
        )

        cv2.circle(
            frame,
            (bx, by),
            5,
            (0, 255, 0),
            -1
        )

        # Draw line between person and bag
        cv2.line(
            frame,
            (px, py),
            (bx, by),
            (0, 255, 0),
            2
        )

        # Calculate current association score
        # using the same logic as associate()
        person_width = max(1, px2 - px1)
        person_height = max(1, py2 - py1)

        relative_x = (bx - px1) / person_width
        relative_y = (by - py1) / person_height

        position_score = 0.0

        if 0 <= relative_x <= 1.5:
            position_score += 0.5

        if relative_y >= 0.45:
            position_score += 0.5

        # IoB
        ix1 = max(px1, bx1)
        iy1 = max(py1, by1)
        ix2 = min(px2, bx2)
        iy2 = min(py2, by2)

        intersection_width = max(0, ix2 - ix1)
        intersection_height = max(0, iy2 - iy1)

        intersection_area = (
            intersection_width * intersection_height
        )

        bag_area = max(
            1,
            (bx2 - bx1) * (by2 - by1)
        )

        iob = intersection_area / bag_area

        if iob >= self.MIN_IOB:
            iob_score = min(iob, 1.0)
        else:
            iob_score = 0.0

        # Distance
        distance = math.sqrt(
            (px - bx) ** 2 +
            (py - by) ** 2
        )

        if distance <= self.MAX_BAG_DISTANCE:
            distance_score = (
                1 - distance / self.MAX_BAG_DISTANCE
            )
        else:
            distance_score = 0.0

        # Final score
        association_score = (
            0.4 * iob_score +
            0.4 * distance_score +
            0.2 * position_score
        )

        # Label
        label = (
            f"P:{person_id}  "
            f"B:{bag_id}  "
            f"Score:{association_score:.2f}"
        )

        # Put label above person box
        cv2.putText(
            frame,
            label,
            (px1, max(25, py1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

     return frame

    def draw_confirmed_associations_bag_exit(self, frame, detections, bag_detections):
    
        # Go through all confirmed person -> bag associations
         for person_id, bag_id in self.person_bag_associations.items():
    
            person_data = None
            bag_data = None
    
            # Find the corresponding person detection
            for person in detections:
                if person["track_id"] == person_id:
                    person_data = person
                    break
    
            # Find the corresponding bag detection
            for bag in bag_detections:
                if bag["track_id"] == bag_id:
                    bag_data = bag
                    break
    
            # If either object is not currently visible, skip drawing
            if person_data is None or bag_data is None:
                continue
    
            # Person bounding box
            px1, py1, px2, py2 = map(
                int,
                person_data["bbox"]
            )
    
            # Bag bounding box
            bx1, by1, bx2, by2 = map(
                int,
                bag_data["bbox"]
            )
    
            # Draw person box - GREEN
            cv2.rectangle(
                frame,
                (px1, py1),
                (px2, py2),
                (0, 255, 0),
                2
            )
    
            # Draw bag box - GREEN
            cv2.rectangle(
                frame,
                (bx1, by1),
                (bx2, by2),
                (0, 255, 0),
                2
            )
    
            # Get midpoints
            person_midpoint = person_data["bottom_center"]
            bag_midpoint = bag_data["bag_midpoint"]
    
            px, py = map(int, person_midpoint)
            bx, by = map(int, bag_midpoint)
    
            # Draw midpoint points
            cv2.circle(
                frame,
                (px, py),
                5,
                (0, 255, 0),
                -1
            )
    
            cv2.circle(
                frame,
                (bx, by),
                5,
                (0, 255, 0),
                -1
            )
    
            # Draw line between person and bag
            cv2.line(
                frame,
                (px, py),
                (bx, by),
                (0, 255, 0),
                2
            )
    
            # Calculate current association score
            # using the same logic as associate()
            person_width = max(1, px2 - px1)
            person_height = max(1, py2 - py1)
    
            relative_x = (bx - px1) / person_width
            relative_y = (by - py1) / person_height
    
            position_score = 0.0
    
            if 0 <= relative_x <= 1.5:
                position_score += 0.5
    
            if relative_y >= 0.45:
                position_score += 0.5
    
            # IoB
            ix1 = max(px1, bx1)
            iy1 = max(py1, by1)
            ix2 = min(px2, bx2)
            iy2 = min(py2, by2)
    
            intersection_width = max(0, ix2 - ix1)
            intersection_height = max(0, iy2 - iy1)
    
            intersection_area = (
                intersection_width * intersection_height
            )
    
            bag_area = max(
                1,
                (bx2 - bx1) * (by2 - by1)
            )
    
            iob = intersection_area / bag_area
    
            if iob >= self.MIN_IOB:
                iob_score = min(iob, 1.0)
            else:
                iob_score = 0.0
    
            # Distance
            distance = math.sqrt(
                (px - bx) ** 2 +
                (py - by) ** 2
            )
    
            if distance <= self.MAX_BAG_DISTANCE:
                distance_score = (
                    1 - distance / self.MAX_BAG_DISTANCE
                )
            else:
                distance_score = 0.0
    
            # Final score
            association_score = (
                0.4 * iob_score +
                0.4 * distance_score +
                0.2 * position_score
            )
    
            # Label
            label = (
                f"P:{person_id}  "
                f"B:{bag_id}  "
                f"Score:{association_score:.2f}"
            )
    
            # Put label above person box
            cv2.putText(
                frame,
                label,
                (px1, max(25, py1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )
    
         return frame
    