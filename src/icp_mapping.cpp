#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/registration/icp.h>
#include <pcl/filters/voxel_grid.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/statistical_outlier_removal.h>

double getDistance(const pcl::PointXYZI & p){
        return std::sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
    }

bool isApproximatelyEqual(const pcl::PointXYZI& point1, const pcl::PointXYZI& point2, double epsilon = 1e-4) {
    pcl::PointXYZI p;
    p.x = point1.x - point2.x;
    p.y = point1.y - point2.y;
    p.z = point1.z - point2.z;
    return getDistance(p) < epsilon;
}

void statOutliierRemover(const pcl::PointCloud<pcl::PointXYZI>::Ptr & cloud_in, pcl::PointCloud<pcl::PointXYZI>::Ptr & cloud_out){
    /* 
    Statistical Outlier removal: https://pcl.readthedocs.io/projects/tutorials/en/latest/statistical_outlier.html
    The resulting cloud_out contains all points of cloud_in that have an average distance to their 8 nearest neighbors that is below the computed threshold
    Using a standard deviation multiplier of 1.0 and assuming the average distances are normally distributed there is a 84.1% chance that a point will be an inlier
    */
    pcl::StatisticalOutlierRemoval<pcl::PointXYZI> stat_outlier_remover(true);
    stat_outlier_remover.setInputCloud(cloud_in);
    stat_outlier_remover.setMeanK(10);
    stat_outlier_remover.setStddevMulThresh (2);
    stat_outlier_remover.filter(*cloud_out);
}

void downSample(const pcl::PointCloud<pcl::PointXYZI>::Ptr & cloud_in, pcl::PointCloud<pcl::PointXYZI>::Ptr & cloud_out){
    // Downsample pointcloud: https://pcl.readthedocs.io/projects/tutorials/en/latest/voxel_grid.html
    float voxel_leaf_size= 0.02f;
    pcl::VoxelGrid<pcl::PointXYZI> voxel_grid_filter;  
    voxel_grid_filter.setInputCloud(cloud_in);
    voxel_grid_filter.setLeafSize(voxel_leaf_size, voxel_leaf_size, voxel_leaf_size);
    voxel_grid_filter.filter(*cloud_out);
}

void filterPointCloud(const pcl::PointCloud<pcl::PointXYZI>::Ptr & cloud_in, pcl::PointCloud<pcl::PointXYZI>::Ptr & cloud_out){
    // Radius filter: remove points which are inside a sphere of radius `outlier_radius`
    auto condition = [](const pcl::PointXYZI & p) { 
        return getDistance(p) < 1.3;
    };
    cloud_in->erase(std::remove_if(cloud_in->begin(), cloud_in->end(), condition), cloud_in->end());
    statOutliierRemover(cloud_in, cloud_out);
    downSample(cloud_out, cloud_out);
}

int main(int argc, char** argv)
{    
    std::string bag_path = "output.bag";
    rosbag::Bag bag;
    bag.open(bag_path, rosbag::bagmode::Read);

    pcl::PointCloud<pcl::PointXYZI>::Ptr target_cloud(new pcl::PointCloud<pcl::PointXYZI>),
                                         source_cloud(new pcl::PointCloud<pcl::PointXYZI>),
                                         map_cloud(new pcl::PointCloud<pcl::PointXYZI>);

    pcl::IterativeClosestPoint<pcl::PointXYZI, pcl::PointXYZI> icp;
    icp.setMaximumIterations(100);
    icp.setMaxCorrespondenceDistance(1000);
    icp.setTransformationRotationEpsilon(1e-9);
    // icp.setEuclideanFitnessEpsilon(0.0001);

    pcl::visualization::CloudViewer viewer("pointcloud viewer");

   // Get an iterator for the bag
    rosbag::View view(bag, rosbag::TopicQuery("/velodyne_points"));
    rosbag::View::iterator it = view.begin();

    // Eigen::Matrix4f Ti = Eigen::Matrix4f::Identity();

    while (it != view.end())
    {
        // Get the count of the iterator
        size_t count = std::distance(view.begin(), it);
        std::cout << "Iteration: " << count << std::endl;  
        
        sensor_msgs::PointCloud2::ConstPtr cloud_msg = it->instantiate<sensor_msgs::PointCloud2>();
        if (cloud_msg != nullptr)
        {
            pcl::fromROSMsg(*cloud_msg, *source_cloud);
            std::vector<int> indices;
            pcl::removeNaNFromPointCloud(*source_cloud, *source_cloud, indices);
            filterPointCloud(source_cloud, source_cloud);

            if (!target_cloud->empty())
            {
                // Apply ICP
                icp.setInputSource(source_cloud);
                icp.setInputTarget(target_cloud);
                pcl::PointCloud<pcl::PointXYZI> aligned_cloud;
                icp.align(aligned_cloud);

                target_cloud = aligned_cloud.makeShared();

                // Update the target cloud
                *map_cloud += aligned_cloud;
                viewer.showCloud(map_cloud);
            }
            else
            {
                // First iteration, set the source cloud as the target
                target_cloud = source_cloud;
            }
        }
        for (int i = 0; i < 5 && it != view.end(); ++i, ++it);
    }
    bag.close();

    // Save the resulting point cloud to a PCD file
    pcl::io::savePCDFileASCII("resulting_map_ever5.pcd", *map_cloud);
    return 0;
}
