import ee, math

class indices():

	def __init__(self):
		
		
		 
		# list with functions to call for each index
		self.functionList = {"ND_blue_green" : self.ND_blue_green, \
							 "ND_blue_red" : self.ND_blue_red, \
							 "ND_blue_nir" : self.ND_blue_nir, \
							 "ND_blue_swir1" : self.ND_blue_swir1, \
							 "ND_blue_swir2" : self.ND_blue_swir2, \
							 "ND_green_red" : self.ND_green_red, \
							 "ND_green_nir" : self.ND_green_nir, \
							 "ND_green_swir1" : self.ND_green_swir1, \
							 "ND_green_swir2" : self.ND_green_swir2, \
							 "ND_red_swir1" : self.ND_red_swir1, \
							 "ND_red_swir2" : self.ND_red_swir2, \
							 "ND_nir_red" : self.ND_nir_red, \
							 "ND_nir_swir1" : self.ND_nir_swir1, \
							 "ND_nir_swir2" : self.ND_nir_swir2, \
							 "ND_swir1_swir2" : self.ND_swir1_swir2, \
							 "R_swir1_nir" : self.R_swir1_nir, \
							 "R_red_swir1" : self.R_red_swir1, \
							 "EVI" : self.EVI, \
							 "SAVI" : self.SAVI, \
							 "IBI" : self.IBI,\
							 "NBLI" : self.NBLI
							 }


	def addAllTasselCapIndices(self,img): 
		""" Function to get all tasselCap indices """
		
		def getTasseledCap(img):
			"""Function to compute the Tasseled Cap transformation and return an image"""
			
			coefficients = ee.Array([
				[0.3037, 0.2793, 0.4743, 0.5585, 0.5082, 0.1863],
				[-0.2848, -0.2435, -0.5436, 0.7243, 0.0840, -0.1800],
				[0.1509, 0.1973, 0.3279, 0.3406, -0.7112, -0.4572],
				[-0.8242, 0.0849, 0.4392, -0.0580, 0.2012, -0.2768],
				[-0.3280, 0.0549, 0.1075, 0.1855, -0.4357, 0.8085],
				[0.1084, -0.9022, 0.4120, 0.0573, -0.0251, 0.0238]
			]);
		
			bands=ee.List(['blue','green','red','nir','swir1','swir2'])
			
			# Make an Array Image, with a 1-D Array per pixel.
			arrayImage1D = img.select(bands).toArray()
		
			# Make an Array Image with a 2-D Array per pixel, 6x1.
			arrayImage2D = arrayImage1D.toArray(1)
		
			componentsImage = ee.Image(coefficients).matrixMultiply(arrayImage2D).arrayProject([0]).arrayFlatten([['brightness', 'greenness', 'wetness', 'fourth', 'fifth', 'sixth']]).float();
	  
			# Get a multi-band image with TC-named bands.
			return img.addBands(componentsImage);	
			
			
		def addTCAngles(img):

			""" Function to add Tasseled Cap angles and distances to an image. Assumes image has bands: 'brightness', 'greenness', and 'wetness'."""
			
			# Select brightness, greenness, and wetness bands	
			brightness = img.select('brightness');
			greenness = img.select('greenness');
			wetness = img.select('wetness');
	  
			# Calculate Tasseled Cap angles and distances
			tcAngleBG = brightness.atan2(greenness).divide(math.pi).rename(['tcAngleBG']);
			tcAngleGW = greenness.atan2(wetness).divide(math.pi).rename(['tcAngleGW']);
			tcAngleBW = brightness.atan2(wetness).divide(math.pi).rename(['tcAngleBW']);
			tcDistBG = brightness.hypot(greenness).rename(['tcDistBG']);
			tcDistGW = greenness.hypot(wetness).rename(['tcDistGW']);
			tcDistBW = brightness.hypot(wetness).rename(['tcDistBW']);
			img = img.addBands(tcAngleBG).addBands(tcAngleGW).addBands(tcAngleBW).addBands(tcDistBG).addBands(tcDistGW).addBands(tcDistBW);
			
			return img;
	
		img = getTasseledCap(img)
		img = addTCAngles(img)
		return img

	def ND_blue_green(self,img):
		img = img.addBands(img.normalizedDifference(['blue','green']).rename(['ND_blue_green']));
		return img
	
	def ND_blue_red(self,img):
		img = img.addBands(img.normalizedDifference(['blue','red']).rename(['ND_blue_red']));
		return img
	
	def ND_blue_nir(self,img):
		img = img.addBands(img.normalizedDifference(['blue','nir']).rename(['ND_blue_nir']));
		return img
	
	def ND_blue_swir1(self,img):
		img = img.addBands(img.normalizedDifference(['blue','swir1']).rename(['ND_blue_swir1']));
		return img
	
	def ND_blue_swir2(self,img):
		img = img.addBands(img.normalizedDifference(['blue','swir2']).rename(['ND_blue_swir2']));
		return img

	def ND_green_red(self,img):
		img = img.addBands(img.normalizedDifference(['green','red']).rename(['ND_green_red']));
		return img
	
	def ND_green_nir(self,img):
		img = img.addBands(img.normalizedDifference(['green','nir']).rename(['ND_green_nir'])).unitScale(-1.0,0.8);  # NDWBI
		return img
	
	def ND_green_swir1(self,img):
		img = img.addBands(img.normalizedDifference(['green','swir1']).rename(['ND_green_swir1']));  # NDSI, MNDWI
		return img
	
	def ND_green_swir2(self,img):
		img = img.addBands(img.normalizedDifference(['green','swir2']).rename(['ND_green_swir2']));
		return img
		
	def ND_red_swir1(self,img):
		img = img.addBands(img.normalizedDifference(['red','swir1']).rename(['ND_red_swir1']));
		return img
			
	def ND_red_swir2(self,img):
		img = img.addBands(img.normalizedDifference(['red','swir2']).rename(['ND_red_swir2']));
		return img

	def ND_nir_red(self,img):
		img = img.addBands(img.normalizedDifference(['nir','red']).rename(['ND_nir_red']));  # NDVI
		return img
	
	def ND_nir_swir1(self,img):
		img = img.addBands(img.normalizedDifference(['nir','swir1']).rename(['ND_nir_swir1']));  # NDWI, LSWI, -NDBI
		return img
	
	def ND_nir_swir2(self,img):
		img = img.addBands(img.normalizedDifference(['nir','swir2']).rename(['ND_nir_swir2']));  # NBR, MNDVI
		return img

	def ND_swir1_swir2(self,img):
		img = img.addBands(img.normalizedDifference(['swir1','swir2']).rename(['ND_swir1_swir2']));
		return img
  
	def R_swir1_nir(self,img):
		# Add ratios
		img = img.addBands(img.select('swir1').divide(img.select('nir')).rename(['R_swir1_nir']));  # ratio 5/4
		return img
			
	def R_red_swir1(self,img):
		img = img.addBands(img.select('red').divide(img.select('swir1')).rename(['R_red_swir1']));  # ratio 3/5
		return img

	def EVI(self,img):
		#Add Enhanced Vegetation Index (EVI)
		evi = img.expression(
			'2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))', {
			  'NIR': img.select('nir'),
			  'RED': img.select('red'),
			  'BLUE': img.select('blue')
		  }).float();
	
		img = img.addBands(evi.rename(['EVI']).unitScale(-0.1,0.9));

		return img
	
	def NBLI(self,img):
		NBLI= img.normalizedDifference(["red","thermal"]).unitScale(-1,0.25).rename("NBLI")
		return img.addBands(NBLI)
	  
	def SAVI(self,img):
		# Add Soil Adjust Vegetation Index (SAVI)
		# using L = 0.5;
		savi = img.expression(
			'(NIR - RED) * (1 + 0.5)/(NIR + RED + 0.5)', {
			  'NIR': img.select('nir'),
			  'RED': img.select('red')
		  }).float();
		img = img.addBands(savi.rename(['SAVI']).unitScale(-0.2,0.9));

		return img
	  
	def IBI(self,img):
		# Add Index-Based Built-Up Index (IBI)
		ibi_a = img.expression(
			'2*SWIR1/(SWIR1 + NIR)', {
			  'SWIR1': img.select('swir1'),
			  'NIR': img.select('nir')
			}).rename(['IBI_A']);
	

		ibi_b = img.expression(
			'(NIR/(NIR + RED)) + (GREEN/(GREEN + SWIR1))', {
			  'NIR': img.select('nir'),
			  'RED': img.select('red'),
			  'GREEN': img.select('green'),
			  'SWIR1': img.select('swir1')
			}).rename(['IBI_B']);
		
		ibi_a = ibi_a.addBands(ibi_b);
		ibi = ibi_a.normalizedDifference(['IBI_A','IBI_B']);
		img = img.addBands(ibi.rename(['IBI']).unitScale(-0.6,0.2));
		
		return img

	def addTopography(self,img):
		"""  Function to add 30m SRTM elevation and derived slope, aspect, eastness, and 
		northness to an image. Elevation is in meters, slope is between 0 and 90 deg,
		aspect is between 0 and 359 deg. Eastness and northness are unitless and are
		between -1 and 1. """

		# Import SRTM elevation data
		elevation = ee.Image("USGS/SRTMGL1_003").unmask(0);
		
		# Calculate slope, aspect, and hillshade
		topo = ee.Algorithms.Terrain(elevation);
		
		# From aspect (a), calculate eastness (sin a), northness (cos a)
		deg2rad = ee.Number(math.pi).divide(180);
		aspect = topo.select(['aspect']);
		aspect_rad = aspect.multiply(deg2rad);
		eastness = aspect_rad.sin().rename(['eastness']).float();
		northness = aspect_rad.cos().rename(['northness']).float();
		
		# Add topography bands to image
		topo = topo.select(['elevation','slope','aspect']).addBands(eastness).addBands(northness);
		img = img.addBands(topo);
		return img;

	def addJRC(self,img):
		""" Function to add JRC Water layers: 'occurrence', 'change_abs', 
			'change_norm', 'seasonality','transition', 'max_extent' """
		
		jrcImage = ee.Image("JRC/GSW1_0/GlobalSurfaceWater")
		
		img = img.addBands(jrcImage.select(['occurrence']).rename(['occurrence']))
		img = img.addBands(jrcImage.select(['change_abs']).rename(['change_abs']))
		img = img.addBands(jrcImage.select(['change_norm']).rename(['change_norm']))
		img = img.addBands(jrcImage.select(['seasonality']).rename(['seasonality']))
		img = img.addBands(jrcImage.select(['transition']).rename(['transition']))
		img = img.addBands(jrcImage.select(['max_extent']).rename(['max_extent']))
		
		return img


	def getIndices(self,img,covariates):	
		""" add indices to image"""
		
		# no need to add indices that are already there
		indices = self.removeDuplicates(covariates,img.bandNames().getInfo())
		
		for item in indices:
			img = self.functionList[item](img)

		return img


	def removeDuplicates(self,covariateList,bands):
		""" function to remove duplicates, i.e. existing bands do not need to be calculated """
		
		return [elem for elem in covariateList if elem not in bands]

	def renameBands(self,image,prefix):
		'rename bands with prefix'
		
		bandnames = image.bandNames();

		def mapBands(band):
			band = ee.String(prefix).cat('_').cat(band);
			return band;
				
		bandnames = bandnames.map(mapBands)
		
		image = image.rename(bandnames);

		return image;

	def addModis(self,img,year):
		
		start = ee.Date.fromYMD(year,1,1)
		end = ee.Date.fromYMD(year+1,1,1)
	
		auto = ee.Image(ee.ImageCollection("users/servirmekong/autocor").filterDate(start,end).first()).rename(["auto"]).unmask(0)
		img = img.addBands(auto)
		cycle = ee.Image(ee.ImageCollection("users/servirmekong/seasons").filterDate(start,end).first()).unmask(0)
		img = img.addBands(cycle)
		return img

	def addDistCoast(self,img):
		distCoast = ee.Image('projects/servir-mekong/Primitives/DistancetoCoast_1k').float().rename(['distCoast']).unmask(0);
		img = img.addBands(distCoast)
		return img

	def addForest(self,img,year):
		start = ee.Date.fromYMD(year, 1, 1)
		end  = ee.Date.fromYMD(year, 12,31)
		
		tcc = ee.Image(ee.ImageCollection("projects/servir-mekong/yearly_primitives_smoothed/tree_canopy").filterDate(start,end).first()).rename(["tcc"])
		img =img.addBands(tcc)
		treeheight = ee.Image( ee.ImageCollection("projects/servir-mekong/yearly_primitives_smoothed/tree_height").filterDate(start,end).first()).rename(["treeheight"])
		img = img.addBands(treeheight)
		return img


	def addModisSR(self,img,year):

		covariates = ["ND_blue_green","ND_blue_red","ND_blue_nir","ND_blue_swir1","ND_blue_swir2", \
				  "ND_green_red","ND_green_nir","ND_green_swir1","ND_green_swir2","ND_red_swir1",\
				  "ND_red_swir2","ND_nir_red","ND_nir_swir1","ND_nir_swir2","ND_swir1_swir2",\
				  "R_swir1_nir","R_red_swir1","EVI","SAVI","IBI"]
		
		
		start = ee.Date.fromYMD(year, 1, 1)
		end  = ee.Date.fromYMD(year, 3,31)
		mod = ee.Image(ee.ImageCollection("projects/servir-mekong/modisComposites").filterDate(start,end).first()).divide(10000)
		#mod = self.getIndices(mod,covariates)	
		mod = self.renameBands(mod,"jan")
		img =img.addBands(mod)
		
		
		start = ee.Date.fromYMD(year, 4, 1)
		end  = ee.Date.fromYMD(year, 6,30)
		mod = ee.Image(ee.ImageCollection("projects/servir-mekong/modisComposites").filterDate(start,end).first()).divide(10000)
		#mod = self.getIndices(mod,covariates)	
		mod = self.renameBands(mod,"apr")
		img =img.addBands(mod)
		
		start = ee.Date.fromYMD(year, 7, 1)
		end  = ee.Date.fromYMD(year, 9,30)
		mod = ee.Image(ee.ImageCollection("projects/servir-mekong/modisComposites").filterDate(start,end).first()).divide(10000)
		#mod = self.getIndices(mod,covariates)	
		mod = self.renameBands(mod,"jul")
		img =img.addBands(mod)
		
		start = ee.Date.fromYMD(year, 10, 1)
		end  = ee.Date.fromYMD(year, 12,31)
		mod = ee.Image(ee.ImageCollection("projects/servir-mekong/modisComposites").filterDate(start,end).first()).divide(10000)
		#mod = self.getIndices(mod,covariates)	
		mod = self.renameBands(mod,"oct")
		img =img.addBands(mod)
		
		return img		
		
		
		
	def addWater(self,img,y): 
		
		geom = img.geometry()
		jrc = ee.ImageCollection("JRC/GSW1_0/MonthlyHistory")
		start = ee.Date.fromYMD(y, 1, 1)
		end  = ee.Date.fromYMD(y, 12,31)

		if y > 2015:
			start = ee.Date.fromYMD(2014,1,1)
			end = ee.Date.fromYMD(2016,1,1)
				
		jrc = jrc.filterBounds(geom).filterDate(start, end)
		
		
		def getObs(img):
			obs = img.gt(0)
			return img.addBands(obs.rename(['obs']).set('system:time_start', img.get('system:time_start')));  
		
		def getWater(img):
			water = img.eq(2);
			return img.addBands(water.rename(['onlywater']).set('system:time_start', img.get('system:time_start')));  

		totalObs = jrc.map(getObs).select(["obs"]).sum().toFloat()
		totalWater = jrc.map(getWater).select(["onlywater"]).sum().toFloat()
		
		returnTime = totalWater.divide(totalObs).multiply(100).unmask(0)
		
		return img.addBands(ee.Image(returnTime).rename(["water"]))

	#def addOther(self,img):
	#	protected = ee.Image("projects/servir-mekong/staticMaps/protectedArea").rename(["protected"]).unmask(0)
	#	distRoad = ee.Image("projects/servir-mekong/staticMaps/distRoads").rename(["distRoad"]).unmask(0)
	#	distBuildings = ee.Image("projects/servir-mekong/staticMaps/distBuildings").rename(["distBuildings"]).unmask(0)
	#	distStream = ee.Image("projects/servir-mekong/staticMaps/distStream").rename(["distStream"]).unmask(0)
	#	eco = ee.Image("projects/servir-mekong/staticMaps/ecoRegions").rename(["eco"]).unmask(0)
	#	ecoforest = ee.Image("projects/servir-mekong/staticMaps/ecoRegionsForest").rename(["ecoForest"]).unmask(0)
	#	hand = ee.Image("users/gena/GlobalHAND/90m-global/fa").rename(["hand"])
	#	img = img.addBands(protected).addBands(distRoad).addBands(distBuildings).addBands(distStream).addBands(eco).addBands(ecoforest).addBands(hand)
	#	return img

	def addOther(self,img):
		protected = ee.Image("projects/servir-mekong/staticMaps/protectedArea").rename(["protected"])
		distBuildings = ee.Image("projects/servir-mekong/staticMaps/distBuildings").rename(["distBuildings"])
		distStream = ee.Image("projects/servir-mekong/staticMaps/distStream").rename(["distStream"])
		eco = ee.Image("projects/servir-mekong/staticMaps/ecoRegions").rename(["eco"])
		ecoforest = ee.Image("projects/servir-mekong/staticMaps/ecoRegionsForest").rename(["ecoForest"])
		hand = ee.Image("users/gena/GlobalHAND/90m-global/fa").rename(["hand"])
		roadDistPrimary = ee.Image("projects/servir-mekong/staticMaps/primaryRoads").rename("primaryRoads")
		roadDistSecondary = ee.Image("projects/servir-mekong/staticMaps/secondaryRoads").rename("secondaryRoads")
		streamDist = ee.Image("WWF/HydroSHEDS/15ACC").rename("stream").unmask(0)
		intAirport = ee.Image("projects/servir-mekong/staticMaps/InternationalAirportDist").rename("intAirport").unmask(0)
		domAirport = ee.Image("projects/servir-mekong/staticMaps/domesticAirportDist").rename("domestAirport").unmask(0)
		coastline = ee.Image("projects/servir-mekong/staticMaps/coastLine").rename("coastline").unmask(0)
		primaryForest = ee.Image("projects/servir-mekong/staticMaps/primaryForest").rename("primaryForest").unmask(0)
		phone = ee.Image("projects/servir-mekong/temp/phoneDataKriging").rename("phone").unmask(0)
		accu = ee.Image("WWF/HydroSHEDS/30ACC").rename("accu")
		countryCode = ee.Image("projects/servir-mekong/staticMaps/countryCode").rename("countryCode")
		power = ee.Image("projects/servir-mekong/staticMaps/powerStations").rename("power")
		forestRotations = ee.Image("projects/servir-mekong/staticMaps/forestRotations").rename("rotations")

		tha = ee.Image("projects/servir-mekong/staticMaps/THA_births_pp_v1_2015")
		vnm = ee.Image("projects/servir-mekong/staticMaps/VNM_births_pp_v2_2015")
		cam = ee.Image("projects/servir-mekong/staticMaps/KHM_births_pp_v2_2015")
		lao = ee.Image("projects/servir-mekong/staticMaps/LAO_births_pp_v2_2015")
		mym = ee.Image("projects/servir-mekong/staticMaps/MMR_births_pp_v2_2015")

		births = ee.Image(ee.ImageCollection([tha,vnm,cam,lao,mym]).mean()).unmask(0).rename("births")


		img = img.addBands(protected).addBands(distBuildings).addBands(distStream).addBands(eco).addBands(ecoforest).addBands(hand).addBands(roadDistPrimary)\
				 .addBands(roadDistSecondary).addBands(streamDist).addBands(intAirport).addBands(domAirport).addBands(coastline).addBands(primaryForest).addBands(phone)\
				 .addBands(accu).addBands(countryCode).addBands(power).addBands(forestRotations).addBands(births)
		return img
		
def returnCovariates(img,year):
	
	
	# hard coded for now
	bands = ['blue','green','red','nir','swir1', 'swir2',"thermal"]	
	bandLow = ['blue_p20','green_p20','red_p20','nir_p20','swir1_p20', 'swir2_p20','thermal_p20']
	bandHigh = ['blue_p80','green_p80','red_p80','nir_p80','swir1_p80', 'swir2_p80','thermal_p80']

	"""Calculate the urban, builtup cropland rice and barren primitives """
	covariates = ["ND_blue_green","ND_blue_red","ND_blue_nir","ND_blue_swir1","ND_blue_swir2", \
				  "ND_green_red","ND_green_nir","ND_green_swir1","ND_green_swir2","ND_red_swir1",\
				  "ND_red_swir2","ND_nir_red","ND_nir_swir1","ND_nir_swir2","ND_swir1_swir2",\
				  "R_swir1_nir","R_red_swir1","EVI","SAVI","IBI","NBLI"]
		
	index = indices()
		
	def addIndices(img,prefix):
		img = img.divide(10000)
		#image = scaleBands(img)
		#img = index.addAllTasselCapIndices(img)
		img = index.getIndices(img,covariates)
		if len(prefix) > 0:	
			img = index.renameBands(img,prefix)
		#else:
			#img = index.addJRC(img).unmask(0)
			#img = index.addTopography(img).unmask(0)	
			#img = index.addModis(img,year).unmask(0)
			#img = index.addModisSR(img,year).unmask(0)
			#img = index.addDistCoast(img).unmask(0)
			#img = index.addForest(img,year).unmask(0)
			#img = index.addWater(img,year).unmask(0)
			#img = index.addOther(img).unmask(0)
	
		return img
			
			
		
	down = addIndices(img.select(bandLow,bands),"p20")
	middle = addIndices(img.select(bands),"")
	up = addIndices(img.select(bandHigh,bands),"p80")
		
	img = down.addBands(middle).addBands(up)
	return img



	
if __name__ == "__main__":
	
	ee.Initialize()
	import numpy as np

	xs = 128 # size of the array
	folder = "waterCNN128"
	bands = ['ND_green_nir',"red","green","blue","nir","swir1","swir2"]
	years = [2019]
	for y in years: #range(2014,2019,1):
		print(y)

		startDate = ee.Date.fromYMD(y,1,1)
		endDate = ee.Date.fromYMD(y,12,31)
		
		training = ee.FeatureCollection("projects/servir-mekong/Temp/waterCNN")
		
		myList = ee.List.repeat(1, xs)
		myLists = ee.List.repeat(myList, xs)
		kernel = ee.Kernel.fixed(xs, xs,myLists)

		print("getting the image")	
			
		img = returnCovariates(ee.Image(ee.ImageCollection("projects/servir-mekong/regionalComposites").filterDate(startDate,endDate).first()).unmask(0),y).select(bands)
				
		water = ee.Image(ee.ImageCollection("projects/servir-mekong/osm/osmWater").mean()).gt(0).rename("class")
			
		img = img.addBands(water).float()
		neighborhood = img.neighborhoodToArray(kernel);
		
		s = training.size().getInfo()
		myNumbers = np.array(range(0,s,1))
		counter = 0
		
		for i in myNumbers:
			print s
			trainingSample = ee.Feature(training.toList(500).get(counter))

			
			counter += 1
			
				
			trainingData = neighborhood.sampleRegions(collection=trainingSample,scale=30,tileScale=16)
				
			trainFilePrefix = folder+'/training_' +str (y)+str(counter)
				
			bands = img.bandNames().getInfo()
						
		
			featureNames = list(bands)
		
			outputBucket = 'servirmekong'
			
				
			trainingTask = ee.batch.Export.table.toCloudStorage(
					collection=trainingData,
					description='TrainingExport'+ str(y)+str(counter),
					fileNamePrefix=trainFilePrefix,
					bucket=outputBucket,
					fileFormat='TFRecord',
					selectors=featureNames)
				
			trainingTask.start()
		
			
		"""
		testSample = training.filter(ee.Filter.gt("random",0.90))
		testFilePrefix = folder +'/testing_'+str(counter) + str(y)
		testData = neighborhood.sampleRegions(collection=testSample,scale=30,tileScale=16)
		testingTask = ee.batch.Export.table.toCloudStorage(
						collection=testData,
						description='TestingExport' + str(y),
						fileNamePrefix=testFilePrefix,
						#fileNamePrefix='data/' + testFilePrefix,
						bucket=outputBucket,
						fileFormat='TFRecord',
						selectors=featureNames)
			
		# Start the tasks.
		testingTask.start()"""


			

