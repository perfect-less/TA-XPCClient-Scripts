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
AIRPORT_ELEVATION = 101.89464 # m
AIRPORT_LAT = 35.01619097009047 # Deg
AIRPORT_LON = -89.97187689049355 # Deg

ELV_MIN = -20 * deg2rad
ELV_MAX = 10  * deg2rad
THROTTLE_DIVIDER = 100

FEATURE_WINDOW_WIDTH = 5
FEATURE_COLUMNS = ["hralt_m", "theta_rad", "aoac_rad", "cas_mps"]
SEQUENTIAL_LABELS = ['elv_l_rad', 'N1s_rpm']

THRESHOLD_DIST = 10 # km
FINAL_HEIGHT = 30 # m

# Initial Cond.
LONGITUDE_0 = -89.97034510928668 # Deg
LATITUDE_0  = 34.90031746597267 # Deg
ALTITUDE_0  = 540 + AIRPORT_ELEVATION # m

THETA_0 = 3.3397455223670303 # Deg
TRUE_HEADING_0 = 0 # Deg 0.09416212074411674

GROUND_SPEED_0 = 80.5 # mps
THROTTLE_0 = .6 # [0-1]

def Elv2LonStick(elv_def_deg: float) -> float:
    if elv_def_deg >= 0:
        lon_stick = elv_def_deg / ELV_MAX
    else:
        lon_stick = elv_def_deg / ELV_MIN

    return lon_stick

def run(model: tf.keras.Model, norm_param):

    lastUpdate = datetime.datetime.now()

    print ("Begin establishing connection with X Plane")
    with xpc.XPlaneConnect() as client:
        # Verify connection
        try:
            # If X-Plane does not respond to the request, a timeout error
            # will be raised.
            client.getDREF("sim/test/test_float")
        except:
            print("Error establishing connection to X-Plane.")
            print("Exiting...")
            return


        # Initial Setup

        time = 0
        last_height = 9999
        rec = []

        # Set aircraft initial position
        print("Setting position")
        #       Lat         Lon          Alt         Pitch Roll True_Heading    Gear
        posi_init = [LATITUDE_0, LONGITUDE_0, ALTITUDE_0, 0,    0,   TRUE_HEADING_0, 0]
        client.sendPOSI(posi_init)

        # Set initial control
        #       Lat Lon Alt Throttle    Gear Flaps
        ctrl_init = [0,  0,  0,  THROTTLE_0, 0,   0]
        client.sendCTRL(ctrl_init)

        # Set Data [EXAMPLE] Setting x plane param
        drefs_init = [
                    'sim/flightmodel/position/local_vy', # mps
                    'sim/flightmodel/position/local_vz',

                ]
        init_cond = [
                    [-sin(THETA_0*deg2rad) * GROUND_SPEED_0],
                    [ cos(THETA_0*deg2rad) * GROUND_SPEED_0]
                ]
        client.sendDREFs(drefs_init, init_cond)


        # Pause the sim
        print("Pausing")
        client.pauseSim(True)
        sleep(2)

        # Toggle pause state to resume
        print("Resuming")
        client.pauseSim(False)


        # Control Loop
        while last_height > FINAL_HEIGHT:
            if (datetime.datetime.now() - lastUpdate).total_seconds() >= INTERVAL:


                # Getting data from x plane
                drefs_get = [
                            'sim/flightmodel/position/alpha',                                   # Deg
                            'sim/flightmodel/position/theta',                                   # Deg
                            'sim/cockpit2/gauges/indicators/calibrated_airspeed_kts_pilot',     # Knots
                            'sim/cockpit2/gauges/indicators/radio_altimeter_height_ft_pilot',   # Feet
                            'sim/flightmodel/controls/elv1_def'                                 # Deg
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
                hralt = dref_values[3][0] * 0.3048    # m
                cas = dref_values[2][0] * 0.514444    # mps
                elv = dref_values[4][0] * deg2rad     # Rad
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
                if dist >= THRESHOLD_DIST:
                    lastUpdate = datetime.datetime.now()
                    continue

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
                feature = np.array(feat_list)
                label = model(feature)

                # Denormalized the labels
                label_send = (label[0, 0], label[0, 1])

                for i, _value in enumerate(label_send):
                    label_send[i] = denorm(_value,
                                            norm_param[SEQUENTIAL_LABELS[i]][0],
                                            norm_param[SEQUENTIAL_LABELS[i]][1]
                                        )
                elv_send, throttle_send = label_send

                elv_send = Elv2LonStick(elv_send * rad2deg )
                throttle_send = throttle_send / THROTTLE_DIVIDER

                # Send labels to x plane
                ctrl_send = [-998, elv_send, -998, throttle_send, -998, -998]
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
