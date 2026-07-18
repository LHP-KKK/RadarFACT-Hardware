# Known limitations

1. The public repository is the hardware/data front end, not the complete
   RadarFACT algorithm release.
2. Hardware synchronization firmware is included but end-to-end trigger-to-
   frame association is still being validated.
3. Older captures used nearest-neighbour radar-camera-LiDAR association.
4. Example pose files from early pipeline tests may be placeholders and must
   not be treated as odometry ground truth.
5. Online radar input, in-memory 4-D FFT, and Jetson-optimized RadarFACT
   inference remain future work.
6. Full DA3/SRT processing can be too heavy for real-time use on an 8 GB Jetson.
7. Exact radar board naming and configurations must be recorded per session.
8. CAD and wiring files are research prototypes, not safety-certified designs.

Before reporting quantitative results, validate synchronization, packet loss,
calibration, point timing, image latency, and sensor temperature stability.

