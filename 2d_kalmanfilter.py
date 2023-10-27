import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
import numpy as np
from numpy.linalg import inv


dt = 1.0/60
F = np.array([[1, dt, 0], [0, 1, dt], [0, 0, 1]])
H = np.array([1, 0, 0]).reshape(1, 3)
Q = np.array([[0.05, 0.05, 0.0], [0.05, 0.05, 0.0], [0.0, 0.0, 0.0]])
R = np.array([0.5]).reshape(1, 1)


class KalmanFilter(Node):
    def __init__(self, F = None, B = None, H = None, Q = None, R = None, P = None, x0 = None):
        super().__init__('kalman_filter_node')
        # Initialize kalman variables

        if(F is None or H is None):
            raise ValueError("Set proper system dynamics.")
        self.estimated_odometry = Odometry()
        self.counter_habda =0 
        self.n = F.shape[1]
        self.m = H.shape[1]

        self.F = F
        self.H = H
        self.B = 0 if B is None else B
        self.Q = np.eye(self.n) if Q is None else Q
        self.R = np.eye(self.n) if R is None else R
        self.P = np.eye(self.n) if P is None else P
        self.x = np.zeros((self.n, 1)) if x0 is None else x0
        # Subscribe to the /odom_noise topic
        self.subscription = self.create_subscription(Odometry,
                                                     '/odom_noise',
                                                     self.odom_callback,
                                                     1)
        
        #publish the estimated reading
        self.estimated_pub=self.create_publisher(Odometry,
                                                 "/odom_estimated",1)

    def predict(self, u = 0):
        self.x = np.dot(self.F, self.x) + np.dot(self.B, u)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self.x

    def update(self, z):
        y = z - np.dot(self.H, self.x)
        S = self.R + np.dot(self.H, np.dot(self.P, self.H.T))
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        I = np.eye(self.n)
        self.P = np.dot(np.dot(I - np.dot(K, self.H), self.P), 
        	(I - np.dot(K, self.H)).T) + np.dot(np.dot(K, self.R), K.T)
        


    def odom_callback(self, msg:Odometry):
        # Extract the position measurements from the Odometry message
        position_x = msg.pose.pose._position.x
        # position_y = msg.pose.pose._position.y
        


        # predictions = []
        # predictions.append(np.dot(H,  self.predict())[0])

        predictions = np.dot(H,  self.predict())[0]




        
        # Update step
        self.update(position_x)  

        
        #publish the estimated reading
        self.estimated_odometry.pose.pose.position.x = float(predictions)
        self.estimated_pub.publish(self.estimated_odometry)

        

def main(args=None):
    rclpy.init(args=args)
    node = KalmanFilter(F = F, H = H, Q = Q, R = R)
    rclpy.spin(node)
    rclpy.shutdown()


main()
