"""Put plane on initial position and setup its flight condition
to our defined initial flight condition (altitude, attitude, velocity
, etc.)"""


import datetime

from time import sleep
from math import sin, cos

import numpy as np
import pandas as pd
import tensorflow as tf

import xpc
from simutils import *
from normalization import DF_Nomalize, denorm, _norm



# V V -  -  -  -  -  -  -  -  -  -  V V
# V V          PARAMETERS           V V
# V V -  -  -  -  -  -  -  -  -  -  V V

# flight_31077
INTERVAL = 1 # Seconds
AIRPORT_ELEVATION = 100.279 # 101.89464 # m
AIRPORT_LAT = 35.01619097009047 # Deg
AIRPORT_LON = -89.97187689049355 # Deg

ELV_MIN = -20
ELV_MAX = 10
THROTTLE_DIVIDER = 100

FEATURE_WINDOW_WIDTH = 5
FEATURE_COLUMNS = ["hralt_m", "theta_rad", "aoac_rad", "cas_mps"]
SEQUENTIAL_LABELS = ['elv_l_rad', 'N1s_rpm']

THRESHOLD_DIST = 10 # km
FINAL_HEIGHT = 30 # m

# Initial Cond.
THETA_0 = 0.6262 # Deg
GLIDE_SLOPE = -3.0 # Deg

GROUND_SPEED_0 = 56.52 # mps
THROTTLE_0 = .6 # [0-1]

def Elv2LonStick(elv_def_deg: float) -> float:
    if elv_def_deg >= 0:
        lon_stick = elv_def_deg / ELV_MAX
    else:
        lon_stick = elv_def_deg / ELV_MIN

    return lon_stick

def run(model: tf.keras.Model, norm_param):

    lastUpdate = datetime.datetime.now()
    on_model = False

    # Initial Setup

    time = 0
    last_height = 9999
    rec = []

    print ("Begin establishing connection with X Plane")
    with xpc.XPlaneConnect(timeout=10000) as client:
        # Verify connection
        try:
            # If X-Plane does not respond to the request, a timeout error
            # will be raised.
            client.getDREF("sim/test/test_float")
        except:
            print("Error establishing connection to X-Plane.")
            print("Exiting...")
            return

        # Pause the sim
        print("Pausing")
        client.pauseSim(True)

        # Set initial control
        #       Lat Lon Alt Throttle    Gear Flaps
        ctrl_init = [0,  0,  0,  THROTTLE_0, 0,   1]
        client.sendCTRL(ctrl_init)

        # Set Data [EXAMPLE] Setting x plane param
        drefs_init = [
                    'sim/flightmodel/position/local_vy', # mps
                    'sim/flightmodel/position/local_vz',
                    'sim/flightmodel/position/theta'

                ]
        init_cond = [
                    -sin((THETA_0-GLIDE_SLOPE)*deg2rad) * GROUND_SPEED_0,
                     cos((THETA_0-GLIDE_SLOPE)*deg2rad) * GROUND_SPEED_0,
                    THETA_0
                ]
        client.sendDREFs(drefs_init, init_cond)

        # Toggle pause state to resume
        print("Resuming")
        client.pauseSim(False)

        lastUpdate = datetime.datetime.now()
        sleep(1)


        # Control Loop
        while last_height > FINAL_HEIGHT:
            if (datetime.datetime.now() - lastUpdate).total_seconds() >= INTERVAL:


                # Getting data from x plane
                drefs_get = [
                            'sim/flightmodel/position/alpha',                                   # Deg
                            'sim/flightmodel/position/theta',                                   # Deg
                            'sim/flightmodel/position/true_airspeed',                           # mps
                        #    'sim/cockpit2/gauges/indicators/calibrated_airspeed_kts_pilot',     # Knots
                        #    'sim/cockpit2/gauges/indicators/radio_altimeter_height_ft_pilot',   # Feet
                        #    'sim/flightmodel/controls/elv1_def'                                 # Deg
                            'sim/flightmodel2/wing/elevator1_deg'
                        ]
                dref_values = client.getDREFs(drefs_get)

                posi = client.getPOSI()
                ctrl = client.getCTRL()

                # Process data
                dist = HaversineDistace(
                            posi[0], AIRPORT_LAT,
                            posi[1], AIRPORT_LON
                        )

                alpha = dref_values[0][0] * deg2rad   # Rad
                theta = dref_values[1][0] * deg2rad   # Rad
                hralt = posi[2] -  AIRPORT_ELEVATION #* 0.3048 -    # m
                cas = dref_values[2][0] #*0.514444    # mps
                elv = dref_values[3][8] * deg2rad     # Rad
                throttle = ctrl[3]

                # Record data
                rec.append(
                    (time, hralt, theta, alpha, cas, elv, throttle)
                )
                print("%1f s, Alti: %2f, Att: (%4f, %4f) CAS:%2f Elevator:%2f Throttle:%2f"\
                        % (time, hralt, theta, alpha, cas, elv, throttle),
                        end='\r')

                # Send back data to x plane
                time += INTERVAL
                if not on_model and (dist >= THRESHOLD_DIST or len (rec) < FEATURE_WINDOW_WIDTH):
                    lastUpdate = datetime.datetime.now()
                    continue
                else:
                    on_model = True
                # continue
                #
                #
                feat_list = []
                for i in range(FEATURE_WINDOW_WIDTH):
                    # Get features from recording
                    _feat = rec[(-FEATURE_WINDOW_WIDTH+i)][1:(1+len(FEATURE_COLUMNS))]
                    _feat = list(_feat)

                    # Normalized Features
                    for i, value in enumerate(_feat):
                        _column = FEATURE_COLUMNS[i]
                        _feat[i] = _norm(value, norm_param[_column][0], norm_param[_column][1])

                    feat_list.append(_feat)

                # Turn it into array and pass it to model
                feature = np.expand_dims (np.array(feat_list), axis=0)
                label = model(feature).numpy()

                # Denormalized the labels
                label_send = [label[0, 0, 0], label[0, 0, 1]]

                for i, _value in enumerate(label_send):
                    label_send[i] = denorm(_value,
                                            norm_param[SEQUENTIAL_LABELS[i]][0],
                                            norm_param[SEQUENTIAL_LABELS[i]][1]
                                        )
                elv_send, throttle_send = label_send[0], label_send[1]

                elv_send = Elv2LonStick(elv_send * rad2deg)
                throttle_send = throttle_send / THROTTLE_DIVIDER

                # Send labels to x plane
                ctrl_send = [elv_send, 0, -998, throttle_send, -998, -998]
                client.sendCTRL(ctrl_send)

                last_height = hralt
                lastUpdate = datetime.datetime.now()

        # Save recordings into csv
        rec_np = np.array(rec)
        rec_df = pd.DataFrame(rec_np,
                    columns=['time_s',
                            "hralt_m", "theta_rad", "aoac_rad", "cas_mps",
                            'elv_l_rad', 'N1s_rpm'
                            ]
                    )
        rec_df.to_csv("rec.csv")


if __name__ == "__main__":

    # Select Model
    model_path = SelectModelPrompt('Models')

    # Import model
    model, _ = LoadModel(model_path)

    # Import train_DF to obtain normalization param.
    train_DF = pd.read_csv('Train_set.csv')
    train_DF, norm_param = DF_Nomalize(train_DF)

    # Run simulation
    run(model, norm_param)

    #
    print ('Simulation Finished')
