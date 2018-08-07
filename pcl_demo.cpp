#include <iostream>
#include <string>

#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/ndt.h>
#include <pcl/filters/approximate_voxel_grid.h>
#include <pcl/filters/voxel_grid_covariance.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/console/time.h>   // TicToc

typedef pcl::PointXYZ PointT;
typedef pcl::PointCloud<PointT> PointCloudT;

bool next_iteration = false;

void print4x4Matrix(const Eigen::Matrix4d & matrix) {
  printf("Rotation matrix :\n");
  printf("    | %6.3f %6.3f %6.3f | \n", matrix(0, 0), matrix(0, 1), matrix(0, 2));
  printf("R = | %6.3f %6.3f %6.3f | \n", matrix(1, 0), matrix(1, 1), matrix(1, 2));
  printf("    | %6.3f %6.3f %6.3f | \n", matrix(2, 0), matrix(2, 1), matrix(2, 2));
  printf("Translation vector :\n");
  printf("t = < %6.3f, %6.3f, %6.3f >\n\n", matrix(0, 3), matrix(1, 3), matrix(2, 3));
}

void keyboardEventOccurred(const pcl::visualization::KeyboardEvent& event,
                            void* nothing) {
  if (event.getKeySym() == "space" && event.keyDown()) {
    std::cout << "space pressed ... ";
    next_iteration = true;
  }
}

int main(int argc, char* argv[]) {
  // The point clouds we will be using
  PointCloudT::Ptr cloud_target(new PointCloudT);
  PointCloudT::Ptr cloud_source_copy(new PointCloudT);  // copy of source cloud
  PointCloudT::Ptr cloud_source(new PointCloudT);  // NDT output point cloud

  // Checking program arguments
  if (argc < 3) {
    printf("Usage :\n");
    printf("\t\t%s source.pcd target.pcd number_of_NDT_iterations\n", argv[0]);
    PCL_ERROR("Provide one pcd file.\n");
    return(-1);
  }

  int iterations = 1;  // Default number of NDT iterations
  if (argc > 3) {
    // If the user passed the number of iteration as an argument
    iterations = atoi(argv[3]);
    if (iterations < 1) {
      PCL_ERROR("Number of initial iterations must be >= 1\n");
      return(-1);
    }
  }

  pcl::console::TicToc time;
  time.tic();

  // load target cloud
  if (pcl::io::loadPCDFile<PointT>(argv[1], *cloud_source) < 0) {
    PCL_ERROR("Error loading cloud %s.\n", argv[1]);
    return(-1);
  }
  std::cout << "\nLoaded source file " << argv[1] << "(" << cloud_source->size() << " points) in "
            << time.toc() << " ms\n" << std::endl;

  // load source cloud
  if (pcl::io::loadPCDFile<PointT>(argv[2], *cloud_target) < 0) {
    PCL_ERROR("Error loading cloud %s.\n", argv[2]);
    return(-1);
  }
  std::cout << "Loaded target file " << argv[2] << "(" << cloud_target->size() << " points) in "
            << time.toc() << " ms\n" << std::endl;

  // extract the voxel grid covar from target cloud
  pcl::VoxelGridCovariance<PointT> covar;
  float leaf_size = 1.0;
  covar.setLeafSize(leaf_size, leaf_size, leaf_size);
  covar.setInputCloud(cloud_target);
  covar.filter(true);

  // get the distribution cloud
  PointCloudT distribution;
  covar.getDisplayCloud(distribution);
  std::string dist_file_path = "dist.pcd";
  pcl::io::savePCDFileASCII(dist_file_path, distribution);
  std::cout << "extracted distribution from target cloud\n";

  Eigen::Matrix4d transformation_matrix = Eigen::Matrix4d::Identity();

  *cloud_source_copy = *cloud_source;  // We backup cloud_source into cloud_source_copy for later use

  PointCloudT::Ptr cloud_filtered_source(new PointCloudT);
  pcl::ApproximateVoxelGrid<PointT> approx_filter;
  approx_filter.setLeafSize(0.2, 0.2, 0.2);
  approx_filter.setInputCloud(cloud_source);
  approx_filter.filter(*cloud_filtered_source);

  std::cout << "filtered source cloud has " << cloud_filtered_source->size()
            << " points\n";

  // Set initial alignment estimate found using robot odometry.
  Eigen::AngleAxisf init_rotation(0.6931, Eigen::Vector3f::UnitZ());
  Eigen::Translation3f init_translation(1.79387, 0.720047, 0);
  Eigen::Matrix4f init_guess =(init_translation * init_rotation).matrix();

  time.tic();
  PointCloudT::Ptr output_cloud(new PointCloudT);

  // The NDT algorithm
  pcl::NormalDistributionsTransform<PointT, PointT> ndt;
  ndt.setMaximumIterations(iterations);
  ndt.setTransformationEpsilon(0.01);
  ndt.setStepSize(0.1);
  ndt.setResolution(1.0);
  ndt.setInputSource(cloud_filtered_source);
  ndt.setInputTarget(cloud_target);
  ndt.align(*output_cloud, init_guess);
//  ndt.align(*cloud_source);
  std::cout << "Applied maximum of " << iterations << " NDT iteration(s) in " << time.toc() << " ms" << std::endl;

  if (ndt.hasConverged()) {
    std::cout << "\nNDT has converged after " << ndt.getFinalNumIteration() << " iterations, score is " << ndt.getFitnessScore() << std::endl;
    std::cout << "\nNDT transformation " << iterations << " : cloud_source -> cloud_target" << std::endl;
    transformation_matrix = ndt.getFinalTransformation().cast<double>();
    print4x4Matrix(transformation_matrix);
    pcl::transformPointCloud(*cloud_filtered_source, *cloud_filtered_source, ndt.getFinalTransformation());
    pcl::transformPointCloud(*cloud_source, *cloud_source, ndt.getFinalTransformation());
    std::string tmp_trans_cloud = "tmp_trans_cloud.pcd";
    pcl::io::savePCDFileASCII(tmp_trans_cloud, *cloud_source);
  } else {
    PCL_ERROR("\nNDT has not converged.\n");
    return(-1);
  }

  std::cout << "\n*********************** visualization step ********************************\n";

  // Visualization
  pcl::visualization::PCLVisualizer viewer("NDT demo");
  // Create two vertically separated viewports
  int v1(0);
  int v2(1);
  viewer.createViewPort(0.0, 0.0, 0.5, 1.0, v1);
  viewer.createViewPort(0.5, 0.0, 1.0, 1.0, v2);

  // The color we will be using
  float bckgr_gray_level = 0.0;  // Black
  float txt_gray_lvl = 1.0 - bckgr_gray_level;

  // target point cloud is white
  pcl::visualization::PointCloudColorHandlerCustom<PointT> cloud_in_color_h(cloud_target,(int) 255 * txt_gray_lvl,(int) 255 * txt_gray_lvl,
                                                                            (int) 255 * txt_gray_lvl);
  viewer.addPointCloud(cloud_target, cloud_in_color_h, "cloud_in_v1", v1);
  viewer.addPointCloud(cloud_target, cloud_in_color_h, "cloud_in_v2", v2);

  // copy of source point cloud is green
  pcl::visualization::PointCloudColorHandlerCustom<PointT> cloud_copy_source_color_h(cloud_source_copy, 20, 180, 20);
  viewer.addPointCloud(cloud_source_copy, cloud_copy_source_color_h, "cloud_tr_v1", v1);

  // NDT aligned source point cloud is red
  pcl::visualization::PointCloudColorHandlerCustom<PointT> cloud_source_color_h(cloud_source, 180, 20, 20);
  viewer.addPointCloud(cloud_source, cloud_source_color_h, "cloud_icp_v2", v2);

  // Adding text descriptions in each viewport
  viewer.addText("White: Target point cloud\nGreen: Copy of source point cloud", 10, 15, 16, txt_gray_lvl, txt_gray_lvl, txt_gray_lvl, "icp_info_1", v1);
  viewer.addText("White: Target point cloud\nRed: NDT aligned source point cloud", 10, 15, 16, txt_gray_lvl, txt_gray_lvl, txt_gray_lvl, "icp_info_2", v2);

  std::stringstream ss;
  ss << iterations;
  std::string iterations_cnt = "NDT iterations = " + ss.str();
  viewer.addText(iterations_cnt, 10, 60, 16, txt_gray_lvl, txt_gray_lvl, txt_gray_lvl, "iterations_cnt", v2);

  // Set background color
  viewer.setBackgroundColor(bckgr_gray_level, bckgr_gray_level, bckgr_gray_level, v1);
  viewer.setBackgroundColor(bckgr_gray_level, bckgr_gray_level, bckgr_gray_level, v2);

  // Set camera position and orientation
  viewer.setCameraPosition(-3.68332, 2.94092, 5.71266, 0.289847, 0.921947, -0.256907, 0);
  viewer.setSize(1280, 1024);  // Visualiser window size

  // Register keyboard callback :
  viewer.registerKeyboardCallback(&keyboardEventOccurred,(void*) NULL);

  transformation_matrix = Eigen::Matrix4d::Identity();
  // use the transformed filtered source cloud
  pcl::NormalDistributionsTransform<PointT, PointT> ndt2;
  ndt2.setMaximumIterations(1);
  ndt2.setTransformationEpsilon(0.01);
  ndt2.setStepSize(0.1);
  ndt2.setResolution(1.0);
  ndt2.setInputSource(cloud_filtered_source);
  ndt2.setInputTarget(cloud_target);
  output_cloud = boost::shared_ptr<PointCloudT> (new PointCloudT);


  // Display the visualiser
  while (!viewer.wasStopped()) {
    viewer.spinOnce();

    // The user pressed "space" :
    if (next_iteration) {
      // The Iterative Closest Point algorithm
      time.tic();
      ndt2.align(*output_cloud);
      std::cout << "Applied 1 NDT iteration in " << time.toc() << " ms" << std::endl;

      if (ndt2.hasConverged()) {
//        printf("\033[11A");  // Go up 11 lines in terminal output.
        printf("\nNDT has converged after %d iterations, score is %+.0e\n", ndt2.getFinalNumIteration(), ndt2.getFitnessScore());
        std::cout << "\nNDT transformation " << ++iterations << " : cloud_source -> cloud_target" << std::endl;
        transformation_matrix *= ndt2.getFinalTransformation().cast<double>();  // WARNING /!\ This is not accurate! For "educational" purpose only!
        print4x4Matrix(transformation_matrix);  // Print the transformation between original pose and current pose

        ss.str("");
        ss << iterations;
        std::string iterations_cnt = "NDT iterations = " + ss.str();
        viewer.updateText(iterations_cnt, 10, 60, 16, txt_gray_lvl, txt_gray_lvl, txt_gray_lvl, "iterations_cnt");
        // have to transform the source cloud manually
        pcl::transformPointCloud(*cloud_filtered_source, *cloud_filtered_source, ndt2.getFinalTransformation());
        pcl::transformPointCloud(*cloud_source, *cloud_source, ndt2.getFinalTransformation());
        ndt2.setInputSource(cloud_filtered_source);
        viewer.updatePointCloud(cloud_source, cloud_source_color_h, "cloud_icp_v2");
      } else {
        PCL_ERROR("\nNDT has not converged.\n");
        return(-1);
      }
    }
    next_iteration = false;
  }
  pcl::io::savePCDFileASCII("transformed_scan1.pcd", *cloud_source);
  return 0;
}