from collections import deque
from gate.gate_manager_bagperson import GateManagerBag
from utils.geometry import get_side


class PersonWithBagExitValidator:
    def __init__(
            self,
            min_exit_frames=2,
            movement_threshold=10
        ):

        self.MIN_EXIT_FRAMES = min_exit_frames
        self.MOVEMENT_THRESHOLD = movement_threshold

        self.gate_manager = GateManagerBag()

        self.gate_points = self.gate_manager.gate_points
        self.inside_reference = self.gate_manager.inside_reference

        # Temporary person-bag tracking
        # {
        #   person_id:{
        #       bag_id:7,
        #       previous_side:"INSIDE",
        #       crossed:False,
        #       outside_frames:0,
        #       positions:[]
        #   }
        # }

        self.pending_associations = {}
        self.person_with_bag_exit_count = 0

        # Prevent duplicate counting
        self.confirmed_exit_ids = set()

    def evict_lost_tracks(self, lost_ids):
        """Evict departed tracks to prevent memory leaks."""
        if not lost_ids:
            return
        for tid in lost_ids:
            self.pending_associations.pop(tid, None)

    # taking the informations from the association function
    def update_associations( self, associations, detections ):
        for association in associations:
            person_id = association["person_track_id"]
            bag_id = association["bag_track_id"]
            # Already stored
            if person_id in self.pending_associations:
                continue
            # Find current person position
            for person in detections:
                if person["track_id"] == person_id:
                    midpoint = person["center"]
                    current_side = self.get_person_side(midpoint)
                    self.pending_associations[person_id] = {
                        "bag_id": bag_id,
                        # Actual current side
                        "previous_side": current_side,
                        "crossed": False,
                        "outside_frames": 0,
                        "positions": deque(
                            maxlen=10
                        )
                    }
                    break

    # Get person side from gate line
    def get_person_side(self, point):
        if len(self.gate_points) != 2:
            return None
        side = get_side(
            point,
            self.gate_points[0],
            self.gate_points[1],
            self.inside_reference
        )
        return side
  
    # Main update function
    # Call every frame
    def update( self, detections ):
        exit_results = []
        for person in detections:
            person_id = person["track_id"]
          
            if person_id not in self.pending_associations:
                continue
            midpoint = person[ "center" ]
            #retrive the persons details from the pending associations
            data = self.pending_associations[ person_id ]
            #set the midpoint inside the positions in data
            data["positions"].append( midpoint )
            current_side = self.get_person_side( midpoint )
            previous_side = data[ "previous_side" ]

            if ( previous_side == "INSIDE" and current_side == "OUTSIDE" ):
                data["crossed"] = True
            data["previous_side"] = current_side

            # After crossing check movement
            if data["crossed"]:
                moving = self.is_moving_away(data["positions"])
                if moving:
                    data["outside_frames"] += 1

            # Final confirmation
            if ( data["outside_frames"] >= self.MIN_EXIT_FRAMES ):
                if person_id not in self.confirmed_exit_ids:
                    self.person_with_bag_exit_count += 1
                    self.confirmed_exit_ids.add( person_id )

                    exit_results.append({
                        "person_track_id": person_id,
                        "bag_track_id": data["bag_id"],
                        "event": "EXIT_WITH_BAG"
                    })
                    # remove completed event
                    del self.pending_associations[ person_id ]
        return exit_results

    # Check if person moves away from exit line
    def is_moving_away( self, positions ):
        if len(positions) < 2:
            return False
        
        first_point = positions[0]
        last_point = positions[-1]

        movement_y = ( last_point[1] - first_point[1] )
        movement_x = ( last_point[0] - first_point[0] )
        distance = ( movement_x ** 2 + movement_y ** 2 ) ** 0.5

        if distance >= self.MOVEMENT_THRESHOLD:
            return True
        return False

    def get_count(self):
        return self.person_with_bag_exit_count


    def reset(self):
        self.pending_associations.clear()
        self.confirmed_exit_ids.clear()
        self.person_with_bag_exit_count = 0